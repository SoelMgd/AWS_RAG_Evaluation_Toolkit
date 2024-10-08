from setuptools import setup, find_packages

setup(
    name='AWS_RAG_Evaluation_Toolkit',
    version='1.0.0', 
    packages=find_packages(), 
    install_requires=[
        'langchain-community==0.3.0',
        'llama-index-llms-bedrock==0.1.7',
        'llama-index-llms-huggingface==0.1.1',
        'transformers==4.37.2',
        'llama-index-embeddings-huggingface==0.1.1',
        'accelerate==0.27.2',
        'sentence-transformers==3.0.1',
        'unstructured==0.15.12',
        'Markdown==3.7',
        'pandas',
        'numpy',
        'ragas',
        'boto3',
        's3fs',
        'nest_asyncio',
        'datasets',
        'seaborn',
        'ipykernel'
    ],
    include_package_data=True,
    description='Toolkit to evaluate RAG built with llama_index on AWS',
    author='SoelMgd',
    python_requires='>=3.7',  
)
