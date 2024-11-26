# AWS RAG Evaluation Toolkit

This repository provides a toolkit for developers to evaluate RAG (Retrieval-Augmented Generation) models built with `llama_index` on AWS. The toolkit is designed to help you experiment with different configurations of HuggingFace embedding models and LLMs available on AWS BedrockChat. It enables you to quickly iterate using various metrics to identify the most effective configurations for your use case.

## Components

- **`RAGConfig`**: Configure your RAG setup with embedding models, LLMs, and prompts.
- **`RAGEvaluate`**: Evaluate the performance of different configurations using various metrics based on RAGAS.
- **`AnalyzeResults`**: Compare configuration results and visualize distributions of different evaluation metrics.

## Getting Started


1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Add your AWS credentials**:
   Create a file named `aws_credentials.cfg` and add the following content:
    ```ini
    [default]
    aws_access_key_id = YOUR_ACCESS_KEY
    aws_secret_access_key = YOUR_SECRET_KEY
    region_name = YOUR_REGION
    ```

Once everything is set up, follow the example notebooks to create a test set from your documents and test different RAG configurations. This allows you to identify the best configuration for your specific use case (e.g., hallucination reduction, etc.).

## Contributing

Contributions are welcome! If you would like to improve the project, follow these steps:

1. **Fork the repository**
2. **Create a new branch** with a descriptive name for your feature or bugfix:
    ```bash
    git checkout -b feature-name
    ```

3. **Test your changes** to ensure everything works as expected.

5. **Create a pull request** from your branch to the `main` branch of this repository.

---

This toolkit aims to make it easier to evaluate RAG models on AWS. Feel free to reach out with suggestions or questions!



