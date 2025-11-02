## Project Purpose

`epub2pdf` converts fixed-layout EPUB collections—especially manga—into PDFs tailored for large-screen e-readers such as the Sony DPT-RP1. The tool aims to keep page layouts faithful to the original scans while ensuring the resulting PDFs use e-ink friendly margins, paper sizes, and fonts.

## Implementation Overview

- **Calibre Integration**: Uses Calibre’s `ebook-convert` to transform EPUBs into A4 PDFs with device-optimized margins, font embedding, and heuristic cleanup.
- **Image-Only Pipeline**: Automatically detects single-image pages and, when appropriate, switches to an `img2pdf`-driven workflow that lays source images directly onto PDF pages without HTML rendering artifacts.
- **EPUB Inspection**: Analyses spine order, viewport metadata, and image assets to pick the appropriate conversion strategy or provide inspection data for debugging.

## Installation

```bash
uv sync
```

If you prefer `pip`, install Python dependencies manually:

```bash
pip install img2pdf pillow
```

Make sure Calibre is installed and `ebook-convert` is available on your `PATH`. Downloads: <https://calibre-ebook.com/download>

## Usage

### Convert an entire directory (default settings)

```bash
uv run epub2pdf epub_files pdf_output
```

### Force image-only conversion for manga EPUBs

```bash
uv run epub2pdf epub_files --image-only --margin 0 --page-size a4
```

### Convert a single EPUB with custom page size and overwrite existing PDF

```bash
uv run epub2pdf "epub_files/book.epub" \
  --output "pdf_output/book.pdf" \
  --custom-size 200x300 \
  --margin-top 4 --margin-bottom 4 --margin-left 6 --margin-right 6 \
  --force-full-image \
  --overwrite
```

### Inspect the source EPUB before converting

```bash
uv run epub2pdf epub_files --inspect-epub
```

## Development

Ruff manages linting and formatting, and its configuration in `pyproject.toml` mirrors the Zed editor settings: line length 88, Python target version 3.12, lint rules `E`, `W`, `F`, `UP`, `B`, `C4`, `I`, `N`, `SIM` with `E501` ignored, a per-file ignore of `F401` for `__init__.py`, and formatter preferences for double quotes, spaces, and keeping magic trailing commas. Run `uv run ruff check .` and `uv run ruff format .` before opening a pull request.

## License

This project is licensed under the MIT License.
