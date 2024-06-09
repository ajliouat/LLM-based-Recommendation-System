import json
import sys
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import onnxruntime_genai as og
import faiss
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from utils.context_memory import ContextMemory
from utils.embedding_provider import EmbeddingProvider
from utils.faiss_indexer import FaissIndexer
import os

# Set up logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SearchModel:
    """A class to generate reformulated queries based on user input and conversation history."""
    def __init__(self, model_path: str, search_options: Dict[str, Any], max_history_length: int = 5):
        print(f"Model path: {model_path}")
        print(f"Contents of the model path: {os.listdir(model_path)}")
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

def process_user_input(user_input: str, conv_model: SearchModel, embedding_provider: EmbeddingProvider, faiss_indexer: FaissIndexer, products: List[Dict[str, Any]]) -> None:
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

        # Retrieve top matching products
        top_indices = faiss_indexer.search(query_embedding)
        print(f"User query: {user_input}")
        print(f"Reformulated query: {reformulated_query}")
        print("Top matches:")
        for idx in top_indices:
            product = products[idx]
            print(f"  - ID: {product['id']}, Name: {product['name']}, Description: {product['description']}")

    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")

def main():
    """The main function that orchestrates the search process."""
    # Load search options and product data
    search_options_path = '/app/search_options.json'
    product_data_path = '/app/product_data.json'
    search_options = load_search_options(search_options_path)
    products = load_product_data(product_data_path)
    product_descriptions = [product['description'] for product in products]

    # Initialize the model and components

    try:
        model_path = '/app/cpu_and_mobile/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4'
        model = og.Model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")


    model_path = '/app/cpu_and_mobile/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4'
    conv_model = SearchModel(model_path, search_options, max_history_length=5)
    embedding_provider = EmbeddingProvider('all-mpnet-base-v2')
    product_embeddings = np.array([embedding_provider.get_embedding(desc) for desc in product_descriptions])
    faiss_indexer = FaissIndexer(embedding_size=product_embeddings.shape[1])
    faiss_indexer.add_items(product_embeddings)

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

            executor.submit(process_user_input, user_input, conv_model, embedding_provider, faiss_indexer, products)

if __name__ == "__main__":
    main()