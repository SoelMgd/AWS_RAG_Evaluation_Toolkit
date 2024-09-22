# AWS RAG Evaluation Toolkit

This repository provides a toolkit for developers to evaluate RAG (Retrieval-Augmented Generation) models built with `llama_index` on AWS. The toolkit is designed to help you experiment with different configurations of HuggingFace embedding models and LLMs available on AWS BedrockChat. It enables you to quickly iterate using various metrics to identify the most effective configurations for your use case.

## Components

- **`RAGConfig`**: Configure your RAG setup with embedding models, LLMs, and prompts.
- **`RAGEvaluate`**: Evaluate the performance of different configurations using various metrics based on RAGAS.
- **`AnalyzeResults`**: Compare configuration results and visualize distributions of different evaluation metrics.

## Getting Started

1. **Create a virtual environment**:
    ```bash
    python -m venv dev_environment
    ```

2. **Activate the virtual environment**:
    - On macOS/Linux:
      ```bash
      source dev_environment/bin/activate
      ```
    - On Windows:
      ```bash
      source dev_environment/Scripts/activate
      ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Add your AWS credentials**:
   Create a file named `aws_credentials.cfg` and add the following content:
    ```ini
    [default]
    aws_access_key_id = YOUR_ACCESS_KEY
    aws_secret_access_key = YOUR_SECRET_KEY
    region_name = YOUR_REGION
    ```

Once everything is set up, follow the example notebooks to create a test set from your documents and test different RAG configurations. This allows you to identify the best configuration for your specific use case (e.g., hallucination reduction, etc.).


