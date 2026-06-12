import os

from langchain_ollama import ChatOllama


class OllamaLLM:
    """
    Ollama LLM Provider

    Expected .env:

        OLLAMA_URL=http://localhost:11434
        OLLAMA_LLM_MODEL=llama3.2

    Examples:

        llama3.2
        llama3.1:8b
        mistral
        qwen3:8b
        gemma3
    """

    def __init__(self):
        self.base_url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434"
        )

        self.model = os.getenv(
            "OLLAMA_LLM_MODEL",
            "llama3.2"
        )

        self.temperature = float(
            os.getenv("TEMPERATURE", "0.2")
        )

        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.
        """

        response = self.llm.invoke(prompt)

        return response.content

    def chat(self, question: str) -> str:
        """
        Simple chat wrapper.
        """

        response = self.llm.invoke(question)

        return response.content

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

Use ONLY the provided context.

If the answer cannot be found,
say:

"I cannot find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content