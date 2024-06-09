import json
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from search_model import SearchModel
from embedding_provider import EmbeddingProvider
from faiss_indexer import FaissIndexer

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

def main():
    """The main function that orchestrates the search process."""
    # Load search options and product data
    search_options_path = 'search_options.json'
    product_data_path = 'product_data.json'
    search_options = load_search_options(search_options_path)
    products = load_product_data(product_data_path)
    product_descriptions = [product['description'] for product in products]

    # Initialize the model and components
    model_path = 'cpu_and_mobile/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4'
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