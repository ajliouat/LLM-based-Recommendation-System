from typing import List

class ContextMemory:
    """A class to manage the conversation history and provide the relevant context."""
    def __init__(self, max_history_length: int = 5):
        self.memory: List[str] = []
        self.max_history_length: int = max_history_length

    def update_memory(self, user_prompt: str) -> None:
        """Updates the conversation history by adding the user prompt and limiting the history length."""
        self.memory.append(f"<|user|>\n{user_prompt}<|end|>")
        if len(self.memory) > self.max_history_length:
            self.memory.pop(0)

    def get_context(self) -> str:
        """Retrieves the conversation history as a string."""
        return "".join(self.memory)