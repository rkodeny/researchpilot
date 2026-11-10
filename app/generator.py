import os
import time
from google.genai.errors import ServerError
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"

def generate_answer(
        query: str,
        retrieved_chunks: list[dict],
)-> str:
    """Generate an answer grounded in retrieved research passages."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not retrieved_chunks:
        return (
            "I couldn't find any relevant information to answer your question."
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your project .env file.")

    client = genai.Client(api_key=api_key)

    evidence_sections = []

    for i,chunk in enumerate(retrieved_chunks, start=1):
        evidence_sections.append(
            f"[Source {i}]\n"
            f"Document: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Passage:\n{chunk['text']}"
        )

    evidence = "\n\n".join(evidence_sections)

    prompt = (
        f"""
            You are ResearchPilot, an academic research assistant.

            Answer the user's question using only the supplied research passages.

            Rules:
                1. Do not invent facts or citations.
                2. If the evidence is insufficient, explicitly say so.
                3. Distinguish what the paper states from your own explanation.
                4. Cite supporting claims using [Source N, p. X], where N
                matches a supplied source and X is its actual page number.
                5. Do not cite a source that does not support the claim.
                6. Explain the answer clearly and concisely.

                User question:
                {query}

                Retrieved research passages:
                {evidence}

                Write the answer with relevant citations.
                """
    )

    response = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            break
        except ServerError as error:
            if attempt == 2:
                raise

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {2 ** (attempt + 1)} seconds..."
            )
            time.sleep(2 ** (attempt + 1))
    if response is None or not response.text:
        raise RuntimeError("The model returned an empty response.")
    return response.text.strip()