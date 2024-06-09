import pytest
from search_model import SearchModel

def test_generate_reformulated_query():
    # Test case 1: Check if the reformulated query is generated correctly
    model_path = 'cpu_and_mobile/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4'
    search_options = {
        "max_length": 1000,
        "num_return_sequences": 1,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 2.0,
        "do_sample": True
    }
    conv_model = SearchModel(model_path, search_options, max_history_length=5)
    user_input = "What are the best electric cars?"
    reformulated_query = conv_model.generate_reformulated_query(user_input)
    assert reformulated_query.startswith("What are")

    # Test case 2: Check if the conversation history is updated correctly
    user_input_2 = "How about affordable electric cars?"
    reformulated_query_2 = conv_model.generate_reformulated_query(user_input_2)
    assert user_input in conv_model.context_memory.get_context()
    assert user_input_2 in conv_model.context_memory.get_context()