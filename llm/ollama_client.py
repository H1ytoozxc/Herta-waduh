from ollama import Client

from config import OllamaConfig


class OllamaChatClient:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self.client = Client(
            host=config.host,
            timeout=config.timeout_seconds,
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat(
            model=self.config.model,
            messages=messages,
            think=self.config.think,
            keep_alive=self.config.keep_alive,
            options={
                "temperature": self.config.temperature,
            },
        )
        return response.message.content.strip()
