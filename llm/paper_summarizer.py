"""Summarize CCUS paper abstracts using an OpenAI-compatible LLM API."""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://api.gapgpt.app/v1"),
    api_key=os.getenv("LLM_API_KEY"),
)

SYSTEM_PROMPT = (
    "You are a CCUS domain expert. Summarize the paper abstract below "
    "in exactly 3 bullet points: (1) material/method, "
    "(2) key quantitative result, (3) main limitation."
)


def summarize_paper(abstract: str, model: str = "gpt-4o-mini") -> str:
    """Return a 3-bullet summary of a CCUS paper abstract."""
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": abstract},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    sample = (
        "We report a novel amine-functionalized MOF for CO2 capture. "
        "The material achieved 3.4 mmol/g uptake at 0.15 bar and 25 C, "
        "with 92% selectivity over N2. Stability decreased after 50 cycles."
    )
    print(summarize_paper(sample))
