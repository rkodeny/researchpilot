import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer
from app.chunking import chunk_pages
from app.ingestion import extract_text_from_pdf 
from app.generator import generate_answer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_vector_index(
        chunks : list[dict],
        model : SentenceTransformer,
):
    """Embed the text chunks and build a FAISS vector index."""

    if not chunks:
        raise ValueError("No text chunks were provided.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy = True,
        normalize_embeddings = True,
        show_progress_bar = True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    #Inner product on normalized vectors equals cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index 

def search_chunks(
        query : str,
        chunks : list[dict],
        index,
        model :SentenceTransformer,
        top_k : int = 5,
) -> list[dict]:
    """Retrieve the most semantically similar text chunks."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not chunks:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_embedding = model.encode(
        [query],
        convert_to_numpy = True,
        normalize_embeddings = True,
    )

    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        result = chunks[int(idx)].copy()
        result["score"] = float(score)
        results.append(result)

    return results

if __name__ == "__main__":
    pdf_path = input("Enter the path to PDF: ").strip()

    try:
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages)

        print(f"\nPages extracted: {len(pages)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"\nLoading embedding model...")

        model = SentenceTransformer(MODEL_NAME)

        print("Building vector index...")
        index = build_vector_index(chunks, model)

        print("\nSemantic search is ready!")
        print("Type 'exit' to stop.")

        while True:
            query = input("\nAsk a question: ").strip()

            if query.lower() == "exit":
                break

            try:
                results = search_chunks(
                    query=query,
                    chunks=chunks,
                    index=index,
                    model=model,
                    top_k = 8,
                )
                print("\nGenerating research answer...")
                answer = generate_answer(query, results)

                print("\n=== ResearchPilot Answer ===\n")
                print(answer)

                print("\n=== Retrieved Evidence ===")


                for rank, result in enumerate(results, start=1):
                    print(f"\n-- Result {rank} | "
                          f"Page: {result['page']} | "
                          f"Score: {result['score']:.3f} --")
                    print(f"Source: {result['source']}")
                    print(result['text'])

            except ValueError as error:
                print(f"Error generating answer: {error}")
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")