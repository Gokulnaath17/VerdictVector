from pathlib import Path

import fitz

from gar.pdf_loader import PDFLoader
from gar.settings import Settings


def make_pdf(path: Path, text: str | None = None) -> None:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_pdf_loader_extracts_text_page(tmp_path: Path):
    pdf = tmp_path / "case.pdf"
    make_pdf(pdf, "This court considered the contract and granted relief.")
    settings = Settings.model_validate({"ocr": {"enabled": False}})
    loader = PDFLoader(settings)

    pages = loader.load(pdf, corpus="case-a")

    assert len(pages) == 1
    assert "contract" in pages[0].text
    assert pages[0].metadata["page"] == 1
    assert pages[0].metadata["extraction_method"] == "text"


def test_pdf_loader_uses_mocked_ocr_for_low_text_page(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    make_pdf(pdf)
    settings = Settings.model_validate(
        {"ocr": {"enabled": True}, "pdf": {"min_text_chars_for_ocr": 10}}
    )
    loader = PDFLoader(settings)
    monkeypatch.setattr(loader, "_ocr_page", lambda page: "OCR extracted legal text")

    pages = loader.load(pdf, corpus="case-a")

    assert pages[0].text == "OCR extracted legal text"
    assert pages[0].metadata["extraction_method"] == "ocr"
