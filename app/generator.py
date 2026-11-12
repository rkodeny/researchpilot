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

    
    prompt = f"""
        You are ResearchPilot, an academic research assistant.

            Answer the user's question using ONLY the supplied research passages.

            STRICT EVIDENCE RULES:

            1. Every factual claim about the research paper must have a citation.
            2. Use citations in this exact format: [Source N, p. X].
            3. N must match the supplied source number.
            4. X must match the actual page number provided for that source.
            5. Before citing a source, verify that its passage directly supports
            the specific claim.
            6. Never invent a source number, page number, quotation, or finding.
            7. If multiple sources support a claim, cite them together.
            8. If the passages do not support a claim, explicitly state that
            the available evidence is insufficient.
            9. Do not present your own interpretation as something the paper
            explicitly states.
            10. Avoid repeating the same evidence in multiple sections.

            ANSWER STRUCTURE:

            - Begin with a direct answer to the question.
            - Organize important findings into clear sections or bullet points.
            - Explain how the findings relate to the question.
            - End with a short conclusion when appropriate.

            USER QUESTION:
            {query}

            RETRIEVED RESEARCH PASSAGES:
            {evidence}

            Write a clear, accurate, evidence-grounded answer.
            """


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

            wait_time = 2 ** (attempt + 1)
            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)

    if response is None or not response.text:
        raise RuntimeError("The model returned an empty response.")
    
    return response.text.strip()