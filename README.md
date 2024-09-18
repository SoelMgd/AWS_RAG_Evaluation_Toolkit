# AWS_RAG_Evaluation_Toolkit


This repository provides a toolkit for developpers to evaluate RAG models built with `llama_index` on AWS.
The toolkit is designed to help you experiment with different configurations of HuggingFace embedding models and LLMs available on AWS BedrockChat and to quickly iterate with metrics to identify the most effective setups for your use case.


## Components

- **`RAGConfig`**: Configure your RAG setup with embedding models, LLMs, and prompts.
- **`RAGEvaluate`**: Evaluate the performance of different configurations using various metrics based on RAGAS.
- **`AnalyzeResults`**: Compare configuration results and display distributions of different metrics.

## Getting Started

1. **Create a venv**

   python -m venv dev_environment

2. **Ativate it**
   source dev_environment/Scripts/activate

3. **Install dependencies**
   pip install -r requirements.txt

4. **Add your AWS Credentials**
   in a aws_credentials.cfg write:
    [default]
    aws_access_key_id = ACCESS_KEY
    aws_secret_access_key = SECRET_KEY
    region_name = REGION

It's done, you can follow the examples notebook to create a testset from your documents and test different RAG configuration on it! Choose the the configuration the most suited to your use-case
(hallucinaton-reduction etc.)
