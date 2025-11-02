import argparse
import io
import logging
import os
import posixpath
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import img2pdf
from PIL import Image, UnidentifiedImageError

Margins = tuple[int, int, int, int]
DEFAULT_BASE_FONT_SIZE = 14
DEFAULT_MARGIN_POINTS = 4
MAX_INSPECTION_PAGES = 5

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def extract_namespace(tag: str) -> dict[str, str]:
    """Return namespace mapping for an XML tag such as '{uri}package'."""
    if tag.startswith("{"):
        uri = tag[1:].split("}", 1)[0]
        return {"opf": uri}
    return {}


@contextmanager
def open_epub_file(zf: zipfile.ZipFile, path: str):
    """Yield a file-like object from the EPUB archive with consistent errors."""
    try:
        with zf.open(path) as stream:
            yield stream
    except KeyError as exc:
        raise ValueError(f"File not found in EPUB: {path}") from exc


@dataclass
class PageInfo:
    html_path: str
    image_sources: list[str]
    has_viewport: bool


@dataclass
class ImagePage:
    html_path: str
    image_path: str
    has_viewport: bool


@dataclass
class EpubAnalysis:
    pages: list[PageInfo]
    image_pages: list[ImagePage]
    is_image_book: bool
    has_viewport: bool


class SingleImageHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.image_sources: list[str] = []
        self.has_viewport = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): value for name, value in attrs if name}
        if tag.lower() == "img":
            src = attributes.get("src")
            if src:
                self.image_sources.append(src)
        elif tag.lower() == "meta":
            if attributes.get("name", "").lower() == "viewport":
                self.has_viewport = True


def find_ebook_convert() -> str:
    """Locate Calibre's ebook-convert executable across supported platforms."""
    candidates: list[str] = []

    env_path = os.environ.get("EBOOK_CONVERT") or os.environ.get(
        "CALIBRE_EBOOK_CONVERT"
    )
    if env_path:
        candidates.append(env_path)

    which_result = shutil.which("ebook-convert")
    if which_result:
        candidates.append(which_result)

    if os.name == "nt":
        program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        program_files_x86 = os.environ.get(
            "PROGRAMFILES(X86)", r"C:\Program Files (x86)"
        )
        local_app_data = os.environ.get("LOCALAPPDATA")
        windows_paths = [
            Path(program_files) / "Calibre2" / "ebook-convert.exe",
            Path(program_files) / "Calibre" / "ebook-convert.exe",
            Path(program_files_x86) / "Calibre2" / "ebook-convert.exe",
        ]
        if local_app_data:
            windows_paths.append(
                Path(local_app_data) / "Calibre2" / "ebook-convert.exe"
            )
        candidates.extend(str(path) for path in windows_paths)
    else:
        unix_paths = [
            "/Applications/calibre.app/Contents/MacOS/ebook-convert",
            "/Applications/Calibre.app/Contents/MacOS/ebook-convert",
            "/usr/bin/ebook-convert",
            "/usr/local/bin/ebook-convert",
            "/opt/calibre/ebook-convert",
        ]
        candidates.extend(unix_paths)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return str(candidate_path)

    raise RuntimeError(
        "ebook-convert not found. Please install Calibre and ensure ebook-convert "
        "is available on PATH: https://calibre-ebook.com/download"
    )


PAPER_SIZES_MM: dict[str, tuple[float, float]] = {
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "b4": (250.0, 353.0),
    "letter": (215.9, 279.4),
    "legal": (215.9, 355.6),
}


def parse_custom_size(custom_size: str | None) -> tuple[float, float] | None:
    if not custom_size:
        return None
    parts = custom_size.lower().replace("mm", "").split("x")
    if len(parts) != 2:
        return None
    try:
        width = float(parts[0])
        height = float(parts[1])
    except ValueError:
        return None
    return width, height


def resolve_page_size(paper_size: str, custom_size: str | None) -> tuple[float, float]:
    custom_mm = parse_custom_size(custom_size)
    if custom_mm:
        return (img2pdf.mm_to_pt(custom_mm[0]), img2pdf.mm_to_pt(custom_mm[1]))
    mm_values = PAPER_SIZES_MM.get(paper_size.lower(), PAPER_SIZES_MM["a4"])
    return (img2pdf.mm_to_pt(mm_values[0]), img2pdf.mm_to_pt(mm_values[1]))


def resolve_zip_href(base_path: str, href: str) -> str:
    normalized = posixpath.normpath(posixpath.join(posixpath.dirname(base_path), href))
    return normalized


def get_spine_document_paths(zf: zipfile.ZipFile) -> list[str]:
    try:
        with open_epub_file(zf, "META-INF/container.xml") as container_file:
            container_xml = container_file.read()
    except ValueError as exc:
        raise ValueError("META-INF/container.xml not found in EPUB") from exc

    container_root = ET.fromstring(container_xml)
    namespace = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = container_root.find("container:rootfiles/container:rootfile", namespace)
    if rootfile is None:
        raise ValueError("EPUB container missing rootfile declaration")
    opf_path = rootfile.attrib.get("full-path")
    if not opf_path:
        raise ValueError("EPUB container missing rootfile path")

    try:
        with open_epub_file(zf, opf_path) as opf_file:
            opf_xml = opf_file.read()
    except ValueError as exc:
        raise ValueError(f"Unable to open OPF package at {opf_path}") from exc

    package_root = ET.fromstring(opf_xml)
    ns = extract_namespace(package_root.tag)

    manifest: dict[str, str] = {}
    manifest_path = "opf:manifest/opf:item" if ns else "manifest/item"
    for item in package_root.findall(manifest_path, ns):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        if item_id and href:
            manifest[item_id] = href

    spine_documents: list[str] = []
    spine_path = "opf:spine/opf:itemref" if ns else "spine/itemref"
    for itemref in package_root.findall(spine_path, ns):
        idref = itemref.attrib.get("idref")
        if not idref:
            continue
        href = manifest.get(idref)
        if not href:
            continue
        doc_path = resolve_zip_href(opf_path, href)
        spine_documents.append(doc_path)

    if spine_documents:
        return spine_documents

    # Fallback: return sorted HTML/XHTML files if spine is empty.
    html_candidates = [
        name
        for name in zf.namelist()
        if name.lower().endswith((".xhtml", ".html", ".htm"))
    ]
    return sorted(html_candidates)


def analyze_epub(epub_path: Path) -> EpubAnalysis:
    with zipfile.ZipFile(epub_path, "r") as zf:
        spine_documents = get_spine_document_paths(zf)
        pages: list[PageInfo] = []
        image_pages: list[ImagePage] = []

        for doc_path in spine_documents:
            try:
                with open_epub_file(zf, doc_path) as doc_file:
                    raw = doc_file.read()
            except ValueError:
                continue
            parser = SingleImageHTMLParser()
            try:
                parser.feed(raw.decode("utf-8"))
            except UnicodeDecodeError:
                parser.feed(raw.decode("utf-8", errors="ignore"))
            parser.close()

            page_info = PageInfo(
                html_path=doc_path,
                image_sources=list(parser.image_sources),
                has_viewport=parser.has_viewport,
            )
            pages.append(page_info)

            if len(parser.image_sources) == 1:
                image_path = resolve_zip_href(doc_path, parser.image_sources[0])
                image_pages.append(
                    ImagePage(
                        html_path=doc_path,
                        image_path=image_path,
                        has_viewport=parser.has_viewport,
                    )
                )

    image_only = bool(image_pages) and all(
        len(page.image_sources) == 1 for page in pages if page.image_sources
    )
    has_viewport = any(page.has_viewport for page in pages)
    return EpubAnalysis(
        pages=pages,
        image_pages=image_pages,
        is_image_book=image_only,
        has_viewport=has_viewport,
    )


def print_epub_inspection(
    epub_path: Path, analysis: EpubAnalysis, limit: int = MAX_INSPECTION_PAGES
) -> None:
    logger.info("Inspecting %s", epub_path)
    if not analysis.image_pages:
        logger.info("No single-image pages detected.")
        return

    sample = analysis.image_pages[:limit]
    with zipfile.ZipFile(epub_path, "r") as zf:
        for page in sample:
            width = height = None
            try:
                with (
                    open_epub_file(zf, page.image_path) as image_stream,
                    Image.open(image_stream) as image,
                ):
                    width, height = image.size
            except Exception as exc:
                logger.error(
                    "Failed to inspect %s -> %s: %s",
                    page.html_path,
                    page.image_path,
                    exc,
                )
                continue
            logger.info(
                "%s -> %s (%sx%s) viewport=%s",
                page.html_path,
                page.image_path,
                width,
                height,
                page.has_viewport,
            )


def convert_images_to_pdf(
    epub_path: Path,
    out_path: Path,
    image_pages: list[ImagePage],
    page_size: tuple[float, float],
    margins: Margins,
    overwrite: bool,
) -> bool:
    if out_path.exists():
        if not overwrite:
            logger.info("Skipping existing file %s", out_path)
            return False
        logger.warning("Overwriting existing file: %s", out_path)

    if not image_pages:
        raise ValueError("No image pages available for image-only conversion")

    top, bottom, left, right = margins
    border = (float(top), float(right), float(bottom), float(left))
    layout_fun = img2pdf.get_layout_fun(page_size)

    buffers: list[io.BytesIO] = []
    with zipfile.ZipFile(epub_path, "r") as zf:
        for index, page in enumerate(image_pages):
            try:
                with open_epub_file(zf, page.image_path) as image_stream:
                    data = image_stream.read()
            except ValueError as exc:
                raise ValueError(f"Missing image asset: {page.image_path}") from exc
            verify_stream = io.BytesIO(data)
            try:
                with Image.open(verify_stream) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError) as exc:
                logger.warning("Skipping invalid image %s: %s", page.image_path, exc)
                continue
            buffer = io.BytesIO(data)
            buffer.name = posixpath.basename(page.image_path) or f"page-{index:03}.jpg"
            buffers.append(buffer)

    if not buffers:
        raise ValueError("No valid images available for image-only conversion")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Running image-only conversion: %s pages -> %s (page=%.1fx%.1fpt, border=%s)",
        len(buffers),
        out_path,
        page_size[0],
        page_size[1],
        border,
    )
    pdf_bytes = img2pdf.convert(buffers, layout_fun=layout_fun, border=border)
    with open(out_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes)
    logger.info(
        "Image-only conversion complete: %s -> %s", epub_path.name, out_path.name
    )
    return True


def build_command(
    converter: str,
    epub_path: Path,
    out_path: Path,
    serif_font: str,
    sans_font: str,
    base_font_size: int,
    paper_size: str,
    custom_size: str | None,
    margins: Margins,
    disable_font_rescaling: bool,
    no_text: bool,
    force_full_image: bool,
) -> list[str]:
    top, bottom, left, right = margins
    command = [
        converter,
        str(epub_path),
        str(out_path),
        "--paper-size",
        paper_size,
        "--output-profile=generic_eink_large",
        "--base-font-size",
        str(base_font_size),
        "--embed-all-fonts",
        "--pdf-serif-family",
        serif_font,
        "--pdf-sans-family",
        sans_font,
        "--pdf-page-margin-top",
        str(top),
        "--pdf-page-margin-bottom",
        str(bottom),
        "--pdf-page-margin-left",
        str(left),
        "--pdf-page-margin-right",
        str(right),
        "--enable-heuristics",
        "--preserve-cover-aspect-ratio",
    ]
    if custom_size:
        command.extend(["--custom-size", custom_size])
    if no_text:
        command.append("--no-text")
    if disable_font_rescaling:
        command.append("--disable-font-rescaling")
    if force_full_image:
        command.extend(
            [
                "--extra-css",
                "img, image, svg, figure { width: 100% !important; height: auto !important; } body { margin: 0 !important; padding: 0 !important; }",
                "--filter-css",
                "width,height,max-width,max-height",
            ]
        )
    return command


def format_command(parts: Iterable[str]) -> str:
    try:
        from shlex import quote
    except ImportError:
        return " ".join(parts)
    return " ".join(quote(part) for part in parts)


def run_calibre_command(cmd: list[str], allow_no_text: bool) -> None:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.stdout:
        logger.info("%s", result.stdout.rstrip())
    if result.stderr:
        logger.warning("%s", result.stderr.rstrip())
    if result.returncode == 0:
        return

    stderr_text = (result.stderr or "").lower()
    if (
        allow_no_text
        and "--no-text" in cmd
        and "no such option: --no-text" in stderr_text
    ):
        logger.warning(
            "--no-text not supported by this Calibre version; retrying without it."
        )
        filtered_cmd = [part for part in cmd if part != "--no-text"]
        logger.info("Retrying command: %s", format_command(filtered_cmd))
        run_calibre_command(filtered_cmd, False)
        return
    if "--filter-css" in cmd and "no such option: --filter-css" in stderr_text:
        logger.warning("--filter-css unsupported; retrying without it.")
        filtered_cmd = []
        skip_next = False
        for part in cmd:
            if skip_next:
                skip_next = False
                continue
            if part == "--filter-css":
                skip_next = True
                continue
            filtered_cmd.append(part)
        logger.info("Retrying command: %s", format_command(filtered_cmd))
        run_calibre_command(filtered_cmd, allow_no_text)
        return

    result.check_returncode()


def convert_epub_to_pdf(
    converter: str,
    epub_path: Path,
    out_path: Path,
    serif_font: str,
    sans_font: str,
    base_font_size: int,
    overwrite: bool,
    paper_size: str,
    custom_size: str | None,
    margins: Margins,
    disable_font_rescaling: bool,
    no_text: bool,
    force_full_image: bool,
) -> bool:
    if out_path.exists():
        if not overwrite:
            logger.info("Skipping existing file %s", out_path)
            return False
        logger.warning("Overwriting existing file: %s", out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        converter=converter,
        epub_path=epub_path,
        out_path=out_path,
        serif_font=serif_font,
        sans_font=sans_font,
        base_font_size=base_font_size,
        paper_size=paper_size,
        custom_size=custom_size,
        margins=margins,
        disable_font_rescaling=disable_font_rescaling,
        no_text=no_text,
        force_full_image=force_full_image,
    )
    logger.info("Running Calibre command: %s", format_command(cmd))
    run_calibre_command(cmd, allow_no_text=no_text)
    logger.info("Calibre conversion complete: %s -> %s", epub_path.name, out_path.name)
    return True


def batch_convert(
    jobs: list[tuple[Path, Path]],
    serif_font: str,
    sans_font: str,
    base_font_size: int,
    overwrite: bool,
    paper_size: str,
    custom_size: str | None,
    margins: Margins,
    disable_font_rescaling: bool,
    no_text: bool,
    force_full_image: bool,
    image_only: bool,
    inspect_epub: bool,
) -> tuple[int, int, int]:
    if not jobs:
        logger.info("No EPUB files found for conversion.")
        return (0, 0, 0)

    converted = 0
    skipped = 0
    failed = 0

    converter: str | None = None
    page_size = resolve_page_size(paper_size, custom_size)

    total_jobs = len(jobs)
    progress = None
    job_iterable = jobs
    if tqdm and total_jobs > 1:
        progress = tqdm(jobs, total=total_jobs, desc="Converting", unit="book")
        job_iterable = progress

    try:
        for index, (epub_path, out_path) in enumerate(job_iterable, 1):
            if progress:
                progress.set_postfix(file=epub_path.name, refresh=False)
            else:
                logger.info("Processing %s/%s: %s", index, total_jobs, epub_path.name)

            analysis: EpubAnalysis | None = None
            use_image_pipeline = False

            if force_full_image or image_only or inspect_epub:
                try:
                    analysis = analyze_epub(epub_path)
                except Exception as exc:
                    logger.warning("Failed to inspect %s: %s", epub_path.name, exc)

            if analysis and inspect_epub:
                print_epub_inspection(epub_path, analysis)

            if analysis:
                if image_only:
                    if analysis.is_image_book and analysis.image_pages:
                        use_image_pipeline = True
                    else:
                        logger.warning(
                            "%s does not appear to be single-image; falling back to Calibre pipeline.",
                            epub_path.name,
                        )
                elif (
                    force_full_image and analysis.is_image_book and analysis.image_pages
                ):
                    logger.info(
                        "Switching to image-only pipeline for %s (fixed-layout detected).",
                        epub_path.name,
                    )
                    use_image_pipeline = True
            elif image_only:
                logger.warning(
                    "Unable to analyze %s; falling back to Calibre pipeline.",
                    epub_path.name,
                )

            if use_image_pipeline and analysis:
                try:
                    if convert_images_to_pdf(
                        epub_path=epub_path,
                        out_path=out_path,
                        image_pages=analysis.image_pages,
                        page_size=page_size,
                        margins=margins,
                        overwrite=overwrite,
                    ):
                        converted += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    failed += 1
                    logger.error(
                        "Image-only conversion failed for %s: %s", epub_path, exc
                    )
                continue

            if converter is None:
                try:
                    converter = find_ebook_convert()
                except RuntimeError as exc:
                    logger.error("%s", exc)
                    failed += 1
                    continue

            try:
                if convert_epub_to_pdf(
                    converter=converter,
                    epub_path=epub_path,
                    out_path=out_path,
                    serif_font=serif_font,
                    sans_font=sans_font,
                    base_font_size=base_font_size,
                    overwrite=overwrite,
                    paper_size=paper_size,
                    custom_size=custom_size,
                    margins=margins,
                    disable_font_rescaling=disable_font_rescaling,
                    no_text=no_text,
                    force_full_image=force_full_image,
                ):
                    converted += 1
                else:
                    skipped += 1
            except subprocess.CalledProcessError as error:
                failed += 1
                logger.error("Conversion failed for %s: %s", epub_path, error)
    finally:
        if progress:
            progress.close()
    return converted, skipped, failed


def resolve_output_path_for_file(
    src_file: Path,
    dst: Path | None,
    output: Path | None,
    default_dir: Path,
) -> Path:
    candidate = output or dst
    if candidate:
        candidate = candidate.resolve(strict=False)
        if candidate.suffix.lower() == ".pdf":
            return candidate
        return (candidate / src_file.stem).with_suffix(".pdf")
    return (default_dir / src_file.stem).with_suffix(".pdf")


def prepare_jobs(
    src: Path,
    dst: Path | None,
    output: Path | None,
    default_dir: Path,
) -> list[tuple[Path, Path]]:
    if not src.exists():
        raise ValueError(f"Source path not found: {src}")
    if src.is_file():
        if src.suffix.lower() != ".epub":
            raise ValueError(f"Source file is not an EPUB: {src}")
        out_path = resolve_output_path_for_file(src, dst, output, default_dir)
        return [(src, out_path)]
    if not src.is_dir():
        raise ValueError(f"Source path is not a directory: {src}")

    destination = (output or dst or default_dir).resolve(strict=False)
    if destination.suffix.lower() == ".pdf":
        raise ValueError(
            "Destination path must be a directory when converting a folder."
        )

    jobs: list[tuple[Path, Path]] = []
    for epub in sorted(src.rglob("*.epub")):
        rel = epub.relative_to(src)
        out_path = (destination / rel).with_suffix(".pdf")
        jobs.append((epub, out_path))
    return jobs


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert EPUB files to A4 PDFs optimized for Sony DPT-RP1."
    )
    parser.add_argument(
        "src",
        type=Path,
        nargs="?",
        default=Path("epub_files"),
        help="Source EPUB directory or file (default: ./epub_files).",
    )
    parser.add_argument(
        "dst",
        type=Path,
        nargs="?",
        default=None,
        help="Destination directory for PDFs (default: ./pdf_output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output path (for file input) or directory override.",
    )
    parser.add_argument(
        "--base-font-size",
        type=int,
        default=DEFAULT_BASE_FONT_SIZE,
        help=f"Base font size in points (default: {DEFAULT_BASE_FONT_SIZE}).",
    )
    parser.add_argument(
        "--paper-size",
        "--page-size",
        dest="paper_size",
        default="a4",
        help="Named paper size for the output PDF (default: a4).",
    )
    parser.add_argument(
        "--custom-size",
        default=None,
        help="Custom PDF page size as WIDTHxHEIGHT in millimeters (overrides paper size).",
    )
    parser.add_argument(
        "--serif-font",
        default="Source Han Serif SC",
        help="PDF serif font family.",
    )
    parser.add_argument(
        "--sans-font",
        default="Source Han Sans SC",
        help="PDF sans-serif font family.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDF files.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=None,
        help="Set all page margins in points (overrides specific margins).",
    )
    parser.add_argument(
        "--margin-top",
        type=int,
        default=DEFAULT_MARGIN_POINTS,
        help=f"Top page margin in points (default: {DEFAULT_MARGIN_POINTS}).",
    )
    parser.add_argument(
        "--margin-bottom",
        type=int,
        default=DEFAULT_MARGIN_POINTS,
        help=f"Bottom page margin in points (default: {DEFAULT_MARGIN_POINTS}).",
    )
    parser.add_argument(
        "--margin-left",
        type=int,
        default=DEFAULT_MARGIN_POINTS,
        help=f"Left page margin in points (default: {DEFAULT_MARGIN_POINTS}).",
    )
    parser.add_argument(
        "--margin-right",
        type=int,
        default=DEFAULT_MARGIN_POINTS,
        help=f"Right page margin in points (default: {DEFAULT_MARGIN_POINTS}).",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Disable text processing to keep original image layout.",
    )
    parser.add_argument(
        "--enable-font-rescaling",
        action="store_true",
        help="Allow Calibre font rescaling (disabled by default).",
    )
    parser.add_argument(
        "--force-full-image",
        action="store_true",
        help="Strip image sizing CSS and force full-width images.",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Use the image-only pipeline instead of Calibre.",
    )
    parser.add_argument(
        "--inspect-epub",
        action="store_true",
        help="Show first-page image details before converting.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv or sys.argv[1:])

    src_path = args.src.resolve(strict=False)
    dst_path = args.dst.resolve(strict=False) if args.dst else None
    output_path = args.output.resolve(strict=False) if args.output else None

    if args.margin is not None:
        args.margin_top = args.margin_bottom = args.margin_left = args.margin_right = (
            args.margin
        )

    margins: Margins = (
        args.margin_top,
        args.margin_bottom,
        args.margin_left,
        args.margin_right,
    )

    default_output_dir = (Path.cwd() / "pdf_output").resolve(strict=False)

    try:
        jobs = prepare_jobs(
            src=src_path,
            dst=dst_path,
            output=output_path,
            default_dir=default_output_dir,
        )
        converted, skipped, failed = batch_convert(
            jobs=jobs,
            serif_font=args.serif_font,
            sans_font=args.sans_font,
            base_font_size=args.base_font_size,
            overwrite=args.overwrite,
            paper_size=args.paper_size,
            custom_size=args.custom_size,
            margins=margins,
            disable_font_rescaling=not args.enable_font_rescaling,
            no_text=args.no_text,
            force_full_image=args.force_full_image,
            image_only=args.image_only,
            inspect_epub=args.inspect_epub,
        )
    except (ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 1

    source_label = src_path if src_path.is_dir() else src_path.name
    logger.info(
        "Summary: converted=%s skipped=%s failed=%s from %s",
        converted,
        skipped,
        failed,
        source_label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
