from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

from gar.documents import PageText
from gar.progress import ProgressCallback, ProgressEvent
from gar.settings import Settings
from gar.utils import content_hash


class PDFLoader:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load(
        self,
        pdf_path: str | Path,
        corpus: str,
        progress: ProgressCallback | None = None,
    ) -> list[PageText]:
        path = Path(pdf_path)
        pages: list[PageText] = []
        with fitz.open(path) as doc:
            total_pages = doc.page_count
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                extraction_method = "text"
                if self._needs_ocr(text):
                    ocr_text = self._ocr_page(page).strip()
                    if ocr_text:
                        text = ocr_text
                        extraction_method = "ocr"

                page_hash = content_hash(text)
                pages.append(
                    PageText(
                        text=text,
                        metadata={
                            "source": str(path),
                            "document": path.name,
                            "corpus": corpus,
                            "page": page_index,
                            "content_hash": page_hash,
                            "extraction_method": extraction_method,
                        },
                    )
                )
                if progress:
                    progress(
                        ProgressEvent(
                            "pdf pages",
                            page_index,
                            total_pages,
                            f"p.{page_index} via {extraction_method}",
                        )
                    )
        return pages

    def _needs_ocr(self, text: str) -> bool:
        return (
            self.settings.ocr.enabled
            and len(text.strip()) < self.settings.pdf.min_text_chars_for_ocr
        )

    def _ocr_page(self, page: fitz.Page) -> str:
        import pytesseract

        if self.settings.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.settings.ocr.tesseract_cmd

        zoom = self.settings.pdf.render_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image, lang=self.settings.ocr.language)
