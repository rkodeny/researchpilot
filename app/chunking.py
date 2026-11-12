from app.ingestion import extract_text_from_pdf
from nltk.tokenize import sent_tokenize



def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """Split Markdown text into sentence-aware chunks, keeping headings attached."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    sections = []
    current_section = []

    for line in text.splitlines():
        if line.startswith("#") and current_section:
            sections.append("\n".join(current_section).strip())
            current_section = []

        current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section).strip())

    chunks = []

    for section in sections:
        if not section:
            continue

        lines = section.splitlines()
        heading = ""

        if lines[0].startswith("#"):
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            body = section

        if not body:
            continue

        sentences = sent_tokenize(body.replace("\n", " "))

        current_chunk = []
        current_length = len(heading) + 1 if heading else 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_chunk and current_length + sentence_length + 1 > chunk_size:
                chunk_text_value = " ".join(current_chunk)
                if heading:
                    chunk_text_value = heading + "\n" + chunk_text_value

                chunks.append(chunk_text_value)

                overlap_sentences = []
                overlap_length = 0

                for previous in reversed(current_chunk):
                    if overlap_length + len(previous) + 1 > overlap:
                        break
                    overlap_sentences.insert(0, previous)
                    overlap_length += len(previous) + 1

                current_chunk = overlap_sentences
                current_length = sum(len(s) + 1 for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length + 1

        if current_chunk:
            chunk_text_value = " ".join(current_chunk)
            if heading:
                chunk_text_value = heading + "\n" + chunk_text_value

            chunks.append(chunk_text_value)

    return chunks


def chunk_pages(
        pages : list[dict],
        chunk_size : int = 1000,
        overlap: int = 150,
) -> list[dict]:
    """chunk extracted PDF pages while preserving source metadata."""

    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size = chunk_size,
            overlap = overlap 
        )

        for chunk_number, text in  enumerate(page_chunks, start = 1):
            all_chunks.append({
                "chunk_id" : len(all_chunks),
                "source" : page["source"],
                "page" : page["page"],
                "chunk_number" : chunk_number,
                "text" : text,
            })    

    return all_chunks

if __name__ == "__main__":
    pdf_path = input("Enter the path to PDF file: ").strip()

    try:
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages)

        print(f"\nPages extracted: {len(pages)}")
        print(f"Chunks created: {len(chunks)}")

        for chunk in chunks[:3]:
            print(f"\n-- Chunk {chunk['chunk_id']} |"
                  f"Page {chunk['page']} ---")
            print(chunk["text"][:300])

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")