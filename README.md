# RAG-Based Recommendation System with Phi-3 Mini-4K-Instruct


[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-latest-blue)](https://www.docker.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.13.1-orange)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red)](https://pytorch.org/)

This project showcases a cutting-edge recommendation system that leverages the state-of-the-art Phi-3 Mini-4K-Instruct model and retrieval-augmented generation (RAG) techniques to deliver highly personalized and context-aware product recommendations. By capturing the user's evolving intent through an iterative process, the system continuously refines its recommendations, ensuring a truly adaptive and engaging user experience.

## 🌟 Key Features

- **Iterative and Evolving Recommendations**: The system employs a sophisticated iterative process to capture and adapt to the user's evolving intent. By incorporating user feedback and interactions, the recommendations are continuously refined, resulting in a highly personalized and dynamic user experience.

- **Phi-3 Mini-4K-Instruct Model**: At the core of the project lies the groundbreaking Phi-3 Mini-4K-Instruct model, a lightweight yet exceptionally capable language model developed by Microsoft. With its ability to understand user intent, generate relevant queries, and provide high-quality recommendations, this model sets a new standard in the field of language understanding and generation.

- **Efficient Inference Optimizations**: To ensure optimal performance and efficiency, the project incorporates several advanced optimization techniques:
  - **Quantization**: The model weights are quantized using INT4 precision, significantly reducing memory footprint and enabling faster inference times without compromising accuracy.
  - **Hardware Acceleration**: By leveraging the power of ONNX Runtime, the project achieves hardware acceleration across various platforms, including CPU, GPU, and mobile devices, enabling seamless deployment and efficient execution.
  - **Embedding Indexing**: The FAISS library is employed for fast similarity search, enabling efficient retrieval of relevant products based on their embeddings. This indexing technique ensures quick and accurate recommendations, even with large product catalogs.

- **Docker Deployment**: The recommendation system is encapsulated within a Docker image, providing a containerized and highly portable solution. This allows for effortless deployment and scalability across diverse environments, making it suitable for a wide range of use cases and infrastructures.

- **Node.js Application**: Accompanying the recommendation system is a user-friendly Node.js application that demonstrates the system's capabilities. It offers an intuitive interface for interacting with the recommendation engine, showcasing the iterative recommendation process and providing a tangible example of the system's potential.

## 🔍 Model Details

The Phi-3 Mini-4K-Instruct model, developed by Microsoft, represents a significant advancement in language modeling. With its lightweight architecture, instruction-tuning, extended context window, and safety-first design, this model is uniquely suited for understanding user intent and generating highly relevant recommendations. Some notable features of the model include:

- **Lightweight Architecture**: Despite its modest size of 3.8 billion parameters, Phi-3 Mini-4K-Instruct achieves remarkable performance, making it ideal for resource-constrained environments and enabling faster inference times without sacrificing quality.

- **Instruction-Tuned**: The model has been fine-tuned to follow a wide range of instructions and communicate effectively, making it particularly adept at understanding user intent and generating appropriate recommendations.

- **Extended Context Window**: With support for a context window of up to 4K tokens, Phi-3 Mini-4K-Instruct can process and reason over longer sequences of text, enabling it to capture the user's evolving intent and consider the conversation history effectively.

- **Safety-First Design**: The model prioritizes safety and responsible AI principles, having undergone rigorous evaluations, including reinforcement learning from human feedback (RLHF) and extensive testing across multiple harm categories.

For a comprehensive understanding of the Phi-3 Mini-4K-Instruct model, please refer to the detailed model card available on Hugging Face: [Phi-3 Mini-4K-Instruct Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

## 🚀 Getting Started

To experience the power of the RAG-based recommendation system firsthand, follow these steps to run the project locally:

1. Clone the repository:
   ```
   git@github.com:ajliouat/rag-based-recommendation-system-with-phi-3-mini-4k-instruct.git
   ```

2. Navigate to the project directory:
   ```
   cd rag-based-recommendation-system
   ```

3. Build the Docker image:
   ```
   docker build -t rag-recommendation .
   ```

4. Run the Docker container:
   ```
   docker run -p 3000:3000 rag-recommendation
   ```

5. Access the Node.js application in your web browser at `http://localhost:3000`.



## 🤝 Contributing

We welcome contributions from the open-source community to further enhance the capabilities of the RAG-based recommendation system. If you encounter any issues, have suggestions for improvements, or would like to contribute new features, please follow the guidelines outlined in [CONTRIBUTING.md](CONTRIBUTING.md).

To contribute to the project, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch from the `main` branch.
3. Make your changes, following the coding conventions and best practices.
4. Write appropriate tests to validate your changes.
5. Commit your changes and push them to your forked repository.
6. Submit a pull request, describing your changes and their benefits.

We appreciate your contributions and will review your pull request as soon as possible.

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE), which allows for open-source use, modification, and distribution. You are free to use, modify, and distribute the software, subject to the conditions specified in the license.

## 🙏 Acknowledgments

We would like to express our gratitude to the following resources and libraries that have been instrumental in the development of this project:

- [Phi-3 Mini-4K-Instruct Model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) by Microsoft, for providing a state-of-the-art language model.
- [ONNX Runtime](https://onnxruntime.ai/), for enabling efficient model inference across various platforms.
- [PyTorch](https://pytorch.org/), for providing a powerful deep learning framework.
- [FAISS](https://github.com/facebookresearch/faiss), for facilitating fast similarity search and efficient embedding indexing.
- [Docker](https://www.docker.com/), for simplifying the deployment and scaling of the recommendation system.

## 📧 Contact

If you have any questions, suggestions, or inquiries regarding the RAG-based recommendation system, please feel free to contact the project maintainer:


Email: [a.jliouat@yahoo.fr](mailto:a.jliouat@yahoo.fr)
GitHub: [ajliouat](https://github.com/ajliouat)

We value your feedback and are committed to continuously improving the recommendation system to meet the evolving needs of our users.
