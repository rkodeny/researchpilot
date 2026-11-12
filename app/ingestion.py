
from pathlib import Path
import pymupdf4llm


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    extracted_pages = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=True
    )

    pages = []

    for page_data in extracted_pages:
        text = page_data["text"].strip()

        if text:
            pages.append({
                "page": page_data["metadata"]["page_number"],
                "text": text,
                "source": path.name,
            })

    return pages


if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ").strip()

    try:
        pages = extract_text_from_pdf(pdf_path)

        print(f"\nExtracted text from {len(pages)} pages.")

        for page in pages[:2]:
            print(f"\n--- PAGE {page['page']} ---")
            print(page["text"][:1000])

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
