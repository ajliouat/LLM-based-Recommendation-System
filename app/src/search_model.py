import onnxruntime_genai as og
from context_memory import ContextMemory
from typing import Dict, Any

class SearchModel:
    """A class to generate reformulated queries based on user input and conversation history."""
    def __init__(self, model_path: str, search_options: Dict[str, Any], max_history_length: int = 5):
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)
        self.search_options = search_options
        self.context_memory = ContextMemory(max_history_length)

    def generate_reformulated_query(self, input_text: str) -> str:
        """Generates a reformulated query based on the user input and conversation history."""
        conversation_history = self.context_memory.get_context()
        self.context_memory.update_memory(input_text)
        full_prompt = f"{conversation_history}<|user|>\nYou're an assistant that helps capture the evolving intent of the user and reformulate on a question. The user's previous history that may be related to the current input:\n{conversation_history}\nThe current user request input is: {input_text}\nReformulate the current input in the form of a question that captures only the recent request of the user. You need to consider only the relevant parts of the previous search history to capture the user's most recent intent. When taking into consideration the previous history of search the most recent one may be more related to the current user prompt.\n<|end|><|user|>\n{input_text}<|end|><|assistant|>"
        input_tokens = self.tokenizer.encode(full_prompt)
        params = og.GeneratorParams(self.model)
        params.set_search_options(**self.search_options)
        params.input_ids = input_tokens
        generator = og.Generator(self.model, params)
        response_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            response_tokens.append(new_token)
        reformulated_query = self.tokenizer.decode(response_tokens)
        reformulated_query = reformulated_query.split('<|assistant|>')[-1].strip()
        return reformulated_query