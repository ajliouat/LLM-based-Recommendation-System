# Define the search options as a dictionary
import json
import os

product_data = [
    {"id": 1, "name": "Tesla Model S", "description": "High-performance electric sedan with long driving range."},
    {"id": 2, "name": "Nissan Leaf", "description": "Affordable and reliable electric hatchback for daily commutes."},
    {"id": 3, "name": "Toyota Camry", "description": "Popular mid-size sedan known for its reliability and comfort."},
    {"id": 4, "name": "Chevrolet Bolt", "description": "Compact electric hatchback with impressive range and technology features."},
    {"id": 5, "name": "Ford Mustang Mach-E", "description": "Electric SUV with sporty performance and sleek design."},
    {"id": 6, "name": "Honda Accord", "description": "Well-rounded mid-size sedan with a spacious interior and fuel efficiency."},
    {"id": 7, "name": "BMW i3", "description": "Compact electric car with a unique design and premium features."},
    {"id": 8, "name": "Audi e-tron", "description": "Luxury electric SUV with advanced technology and refined craftsmanship."},
    {"id": 9, "name": "Hyundai Kona Electric", "description": "Affordable electric SUV with impressive driving range."},
    {"id": 10, "name": "Porsche Taycan", "description": "High-performance electric sports car with cutting-edge technology."}
]

# Specify the folder and file path
file_path = './product_data.json'

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Write the dictionary to a JSON file, overwriting it if it exists
with open(file_path, 'w') as json_file:
    json.dump(product_data, json_file, indent=4)

# Verify that the file has been created and contains the correct content
with open(file_path, 'r') as json_file:
    data = json.load(json_file)
    print(data)


# Define the search options as a dictionary
search_options = {
    "max_length": 2000,
    "num_return_sequences": 1,
    "temperature": 0.5,
    "top_k": 80,
    "top_p": 0.95,
    "repetition_penalty": 2.5,
    "do_sample": True
}

# Specify the folder and file path
file_path = './search_options.json'

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Write the dictionary to a JSON file, overwriting it if it exists
with open(file_path, 'w') as json_file:
    json.dump(search_options, json_file, indent=4)

# Verify that the file has been created and contains the correct content
with open(file_path, 'r') as json_file:
    data = json.load(json_file)
    print(data)


# few_shot_examples
few_shot_examples = [
    {
        "input": "top car companies",
        "output": "What are the leading car brands?"
    },
    {
        "input": "electric vehicles only",
        "output": "What are the top electric vehicle brands?"
    },
    {
        "input": "focus on luxury EVs",
        "output": "What are the best luxury electric vehicle brands?"
    },
    {
        "input": "affordable electric SUVs",
        "output": "What are the most affordable electric SUV models?"
    },
    {
        "input": "compare Tesla and Audi EVs",
        "output": "How do Tesla and Audi electric vehicles compare?"
    }
]

# Specify the folder and file path
file_path = './few_shot_examples.json'

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Write the dictionary to a JSON file, overwriting it if it exists
with open(file_path, 'w') as json_file:
    json.dump(few_shot_examples, json_file, indent=4)

# Verify that the file has been created and contains the correct content
with open(file_path, 'r') as json_file:
    data = json.load(json_file)
    print(data)


# Import libraries
import json
from typing import List, Dict, Any
import numpy as np
import time
import onnxruntime_genai as og
from sentence_transformers import SentenceTransformer
import faiss
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class EmbeddingProvider:
    """A class to generate embeddings for text queries using a pre-trained sentence transformer model."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, query: str) -> np.ndarray:
        """Generates the embedding for a given text query."""
        return self.model.encode(query)

class FaissIndexManager:
    """A class to create and search a FAISS index for efficient similarity search."""
    def __init__(self, embedding_size: int, M: int = 16, ef_construction: int = 200, ef_search: int = 200):
        self.index = faiss.IndexHNSWFlat(embedding_size, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.product_embeddings: List[np.ndarray] = []
        self.product_ids: List[int] = []

    def add_items(self, item_embeddings: np.ndarray, item_ids: List[int]) -> None:
        self.index.add(item_embeddings)
        self.product_embeddings.extend(item_embeddings)
        self.product_ids.extend(item_ids)

    def update_items(self, item_embeddings: np.ndarray, item_ids: List[int]) -> None:
        for item_id, item_embedding in zip(item_ids, item_embeddings):
            if item_id in self.product_ids:
                idx = self.product_ids.index(item_id)
                self.product_embeddings[idx] = item_embedding
            else:
                self.add_items(np.array([item_embedding]), [item_id])
        self.index.reset()
        self.index.add(np.array(self.product_embeddings))

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[int]:
        _, indices = self.index.search(query_embedding, top_k)
        return [self.product_ids[idx] for idx in indices[0]]

class SearchModel:
    """A class to generate reformulated queries based on user input and conversation history."""
    def __init__(self, model_path: str, search_options: Dict[str, Any], few_shot_examples_path: str, product_type: str, max_history_length: int = 5):
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)
        self.search_options = search_options
        self.context_memory = ContextMemory(max_history_length)
        self.few_shot_examples = self.load_few_shot_examples(few_shot_examples_path)
        self.product_type = product_type

    def load_few_shot_examples(self, file_path: str) -> List[Dict[str, str]]:
        with open(file_path, 'r') as file:
            return json.load(file)

    def generate_reformulated_query(self, input_text: str) -> str:
        """Generates a reformulated query based on the user input and conversation history."""
        conversation_history = self.context_memory.get_context()
        self.context_memory.update_memory(input_text)

        few_shot_prompt = ""
        for example in self.few_shot_examples:
            few_shot_prompt += f"<|user|>\n{example['input']}\n<|assistant|>\n{example['output']}\n<|end|>\n"

        if conversation_history.strip():
            context_prompt = f"Conversation History:\n{conversation_history}\n\n"
        else:
            context_prompt = "No previous conversation history.\n\n"

        full_prompt = f"{few_shot_prompt}<|user|>\nAs an AI assistant, your task is to reformulate the user's input into a highly specific question about cars. Focus on the key details provided by the user and generate a question that targets their precise needs or preferences.\n\n{context_prompt}User Input: {input_text}\n\nInstructions:\n- Carefully analyze the user's input and identify the most important details or criteria.\n- Consider the conversation history to understand the context and the user's evolving preferences.\n- Reformulate the input into a clear, concise, and highly specific question that directly addresses the user's needs.\n- Avoid generating broad or generic questions. Focus on the unique aspects mentioned by the user.\n- Use the question to guide the search towards the most relevant car recommendations.\n\n<|end|><|user|>\n{input_text}<|end|><|assistant|>"

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

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Loads a JSON file and returns its contents as a dictionary."""
    with open(file_path, 'r') as file:
        return json.load(file)

def load_product_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads product data from a JSON file."""
    return load_json_file(file_path)

def load_search_options(file_path: str) -> Dict[str, Any]:
    """Loads search options from a JSON file."""
    return load_json_file(file_path)

def process_user_input(user_input: str, conv_model: SearchModel, embedding_provider: EmbeddingProvider, faiss_index_manager: FaissIndexManager, products: List[Dict[str, Any]]) -> None:
    """Processes a single user input and generates a response."""
    start_time = time.time()
    try:
        reformulated_query = conv_model.generate_reformulated_query(user_input)
        query_embedding = embedding_provider.get_embedding(reformulated_query).reshape(1, -1)
        end_time = time.time()
        execution_time = end_time - start_time

        print("Assistant:", reformulated_query)
        print(f"Execution time: {execution_time:.5f} seconds")

        # Print the current state of the conversation history
        print("Current Conversation History:")
        print(conv_model.context_memory.get_context())

        # Retrieve top matching product IDs
        top_product_ids = faiss_index_manager.search(query_embedding)
        print("Top matches:")
        for product_id in top_product_ids:
            product = next(product for product in products if product['id'] == product_id)
            print(f"  - ID: {product['id']}, Name: {product['name']}, Description: {product['description']}")

    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")

def main():
    """The main function that orchestrates the search process."""
    # Load search options, product data, and few-shot examples
    search_options_path = './search_options.json'
    product_data_path = './product_data.json'
    few_shot_examples_path = './few_shot_examples.json'
    search_options = load_search_options(search_options_path)
    products = load_product_data(product_data_path)
    product_descriptions = [product['description'] for product in products]

    # Initialize the model and components
    model_path = './cpu_and_mobile/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4'
    product_type = "car"  # Replace with the appropriate product type
    conv_model = SearchModel(model_path, search_options, few_shot_examples_path, product_type, max_history_length=5)
    embedding_provider = EmbeddingProvider('all-mpnet-base-v2')
    product_embeddings = np.array([embedding_provider.get_embedding(product['description']) for product in products])
    faiss_index_manager = FaissIndexManager(embedding_size=product_embeddings.shape[1])
    faiss_index_manager.add_items(product_embeddings, [product['id'] for product in products])

    # Pre-load models and data
    _ = conv_model.generate_reformulated_query("Dummy input")
    _ = embedding_provider.get_embedding("Dummy query")
    conv_model.context_memory.memory.clear()  # Clear the dummy input from the conversation history

    # Generation loop
    with ThreadPoolExecutor() as executor:
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() == 'exit':
                break

            if not user_input:
                print("Please enter a valid input.")
                continue

            executor.submit(process_user_input, user_input, conv_model, embedding_provider, faiss_index_manager, products)

if __name__ == "__main__":
    main()