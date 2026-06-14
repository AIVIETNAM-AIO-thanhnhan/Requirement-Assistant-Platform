import os

from openai import OpenAI


class OpenAILLM:
    """
    OpenAI LLM Provider

    Expected .env:

        OPENAI_API_KEY=xxxxx
        OPENAI_LLM_MODEL=gpt-4o-mini

    Examples:

        gpt-4o-mini
        gpt-4o
        o4-mini
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.model = os.getenv(
            "OPENAI_LLM_MODEL",
            "gpt-4o-mini"
        )

        self.temperature = float(
            os.getenv("TEMPERATURE", "0.2")
        )

        self.client = OpenAI(
            api_key=api_key
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    def chat(self, question: str) -> str:
        """
        Simple chat wrapper.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return response.choices[0].message.content

    def generate_with_context(
        self,
        context: str,
        question: str
    ) -> str:
        """
        RAG helper method.
        """

        prompt = f"""
You are a QA Documentation Assistant.

Rules:
1. Use ONLY the provided context.
2. Do not invent information.
3. If information is missing, say:
   "I cannot find this information in the uploaded documents."
4. Cite source information when available.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content