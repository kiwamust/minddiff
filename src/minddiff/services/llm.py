"""LLM provider abstraction layer."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base for LLM providers (NFR-06)."""

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        ...


class ClaudeProvider(LLMProvider):
    """Claude API implementation."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
