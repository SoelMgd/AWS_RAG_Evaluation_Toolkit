import json
import pandas as pd
from datasets import Dataset
import asyncio
import boto3
import s3fs

from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings

from llama_index.core import  Settings, StorageContext, load_index_from_storage
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ragas import evaluate


def load_vector_index(s3: s3fs.core.S3FileSystem, s3_bucket_name: str):
    """
    Load the vector index from S3.

    Args:
        s3: S3 object for interacting with Amazon S3.
        s3_bucket_name (str): The name of the S3 bucket.

    Returns:
        loaded_index: The index loaded from storage.
    """
    return load_index_from_storage(
        StorageContext.from_defaults(persist_dir=s3_bucket_name, fs=s3)
    )


def extract_source_info_and_text(source_nodes):
    """
    Extract information and text from source nodes.

    Args:
        source_nodes (list): List of source nodes containing textual information.

    Returns:
        list: List of extracted and cleaned text.
    """
    return [node.node.text.strip() for node in source_nodes]


def get_answer_with_context(question: str, query_engine):
    """
    Retrieve the answer to a question along with its context.

    Args:
        question (str): The question to ask.
        query_engine: Query engine used to execute the question.

    Returns:
        dict: Dictionary containing the answer and the contexts.
    """
    response = query_engine.query(question)
    contexts = extract_source_info_and_text(response.source_nodes)
    
    return {
        'answer': response.response.strip(),
        'contexts': contexts
    }

def load_bedrock_model(model_name: str, session: boto3.session.Session, context_size=None):
    """
    Load a Bedrock model from AWS Bedrock.

    Args:
        session: boto3 session object to handle AWS authentication and configuration.
        model_name (str): The name of the model to load.
        context_size (int, optional): Context size for the model if specified.

    Returns:
        llm: The loaded Bedrock model.
    """
    client = session.client(
        service_name='bedrock-runtime',
        region_name=session.region_name,
        endpoint_url=f"https://bedrock-runtime.{session.region_name}.amazonaws.com"
    )

    return Bedrock(client=client, model=model_name, context_size=context_size)

def prepare_query_engine(config, session: boto3.session.Session,s3: s3fs.core.S3FileSystem, s3_bucket_name: str):
    """
    Prepare the query engine by loading the models and the index.

    Args:
        config (RAGConfig): Configuration object containing model parameters.
        s3: S3 object for interacting with Amazon S3.
        s3_bucket_name (str): The name of the S3 bucket.

    Returns:
        query_engine: Configured query engine based on the provided parameters.
    """
    llm = load_bedrock_model(config.generation_llm, session)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.embedder)
    loaded_index = load_vector_index(s3, s3_bucket_name)
    
    return loaded_index.as_query_engine(response_mode="compact", system_prompt=config.prompt)


def answer_dataset(testset, config, session: boto3.session.Session, s3: s3fs.core.S3FileSystem, s3_bucket_name: str):
    """
    Answer the questions in the test dataset using a RAGConfig.

    Args:
        testset (DataFrame): Dataset containing the questions.
        config: RAG configuration used to generate the answers.
        s3: S3 object for interacting with Amazon S3.
        s3_bucket_name (str): The name of the S3 bucket.

    Returns:
        DataFrame: Dataset with the answers and contexts added.
    """
    query_engine = prepare_query_engine(config, session, s3, s3_bucket_name)
    print("Answering questions\n")

    testset_copy = testset.copy()
    RAG_answer = testset_copy['question'].apply(lambda q: get_answer_with_context(q, query_engine))
    
    testset_copy['RAG_answer'] = RAG_answer.apply(lambda x: x['answer'])
    testset_copy['RAG_contexts'] = RAG_answer.apply(lambda x: x['contexts'])
    
    return testset_copy



class RAGEvaluate:
    """
    Class for evaluating RAG models using different metrics and configurations.
    
    Attributes
    ----------
    config : RAGConfig
        Configuration instance to be used for evaluation.
    s3 : s3fs.core.S3FileSystem
        S3 file system instance for saving/loading data.
    s3_bucket_name : str
        Name of the S3 bucket for storing results and data.
    results : pd.DataFrame or None
        Stores the evaluation results, initialized as None.
    testset_answered : dict or None
        Stores the answered testset, initialized as None.
    """

    def __init__(self, config, session: boto3.session.Session, s3: s3fs.core.S3FileSystem, s3_bucket_name: str):
        """
        Initialize the RAGEvaluate class with provided configuration and S3 bucket.
        
        Parameters
        ----------
        config : RAGConfig
            Configuration for the evaluation.
        s3 : s3fs.core.S3FileSystem
            S3 file system instance.
        s3_bucket_name : str
            Name of the S3 bucket.
        """
        self.config = config
        self.session = session
        self.s3 = s3
        self.s3_bucket_name = s3_bucket_name
        self.results = None
        self.testset_answered = None

    def prepare_scoring_LLM(self, bedrock_client, model_id, temperature=0.4):
        """
        Set up BedrockChat and BedrockEmbeddings for scoring.

        Parameters
        ----------
        bedrock_client : object
            AWS Bedrock client instance.
        model_id : str
            Model ID for BedrockChat.
        temperature : float, optional
            Model temperature for sampling, by default 0.4.

        Returns
        -------
        tuple
            Tuple of (llm4scoring, embeddings).
        """
        llm4scoring = BedrockChat(
            client=bedrock_client, 
            model_id=model_id, 
            model_kwargs={"temperature": temperature}
        )
        embeddings = BedrockEmbeddings(client=bedrock_client)
        return llm4scoring, embeddings

    def answer_testet(self, testset):
        """
        Answer the provided testset and convert it to a RAGAS dataset format.

        Parameters
        ----------
        testset : pd.DataFrame
            The test dataset to answer.

        Returns
        -------
        Dataset
            RAGAS formatted dataset.
        """
        answered_testset = answer_dataset(
            testset, self.config, self.session, self.s3, self.s3_bucket_name
        )

        ragas_dataset = {
            "question": answered_testset['question'].tolist(),
            "ground_truth": answered_testset['ground_truth'].tolist(),
            "contexts": answered_testset["RAG_contexts"].tolist(),
            "answer": answered_testset["RAG_answer"].tolist(),
        }
        return Dataset.from_dict(ragas_dataset)

    async def evaluate_question(self, i, data, metric, llm, embeddings):
        """
        Evaluate a single question using the provided metric, LLM, and embeddings.

        Parameters
        ----------
        i : int
            Question index.
        data : dict
            Question data.
        metric : object
            Metric to use for evaluation.
        llm : object
            LLM for scoring.
        embeddings : object
            Embeddings for scoring.

        Returns
        -------
        dict
            Dictionary with metric name and its corresponding score.
        """
        try:
            temp_dataset = Dataset.from_dict({
                "question": [data["question"]],
                "ground_truth": [data["ground_truth"]],
                "contexts": [data["contexts"]],
                "answer": [data["answer"]]
            })

            score = evaluate(
                temp_dataset, metrics=[metric], llm=llm, embeddings=embeddings
            )
            result = score.to_pandas().iloc[0].to_dict()
            return {metric.name: result[metric.name]}

        except Exception as e:
            print(f"Error evaluating question '{i}' with '{metric.name}': {str(e)}")
            return {metric.name: float('nan')}  # Return NaN for missing values

    async def evaluate_batch(self, batch, metrics, llm4scoring, embeddings):
        """
        Evaluate a batch of questions with the specified metrics.

        Parameters
        ----------
        batch : pd.DataFrame
            Batch of data to evaluate.
        metrics : list
            List of metrics to evaluate.
        llm4scoring : object
            LLM for scoring.
        embeddings : object
            Embeddings for scoring.

        Returns
        -------
        dict
            Dictionary of batch results indexed by question.
        """
        tasks = [
            (i, metric, self.evaluate_question(i, row, metric, llm4scoring, embeddings))
            for i, row in batch.iterrows()
            for metric in metrics
        ]
        
        task_results = await asyncio.gather(*[t[2] for t in tasks])
        batch_results = {}

        for idx, (i, metric, _) in enumerate(tasks):
            if i not in batch_results:
                batch_results[i] = {}
            batch_results[i][metric.name] = task_results[idx].get(metric.name, -1.0)

        return batch_results

    async def evaluate_config(self, config, testset, batch_size=10):
        """
        Evaluate the given configuration on the testset.

        Parameters
        ----------
        config : RAGConfig
            Configuration to evaluate.
        testset : pd.DataFrame
            Test dataset to evaluate.
        batch_size : int, optional
            Batch size for evaluation, by default 10.

        Returns
        -------
        pd.DataFrame, dict
            DataFrame of results and answered testset.
        """
        dataset = self.answer_testet(testset)
        df = dataset.to_pandas()

        bedrock_client = self.session.client(
        service_name='bedrock-runtime',
        region_name= self.session.region_name,
        endpoint_url=f"https://bedrock-runtime.{self.session.region_name}.amazonaws.com")

        llm4scoring, embeddings = self.prepare_scoring_LLM(
            bedrock_client, model_id=config.scoring_llm
        )
        metrics = config.metrics
        
        all_results = {}
        num_batches = (len(df) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(df))
            batch = df.iloc[batch_start:batch_end]
            
            batch_results = await self.evaluate_batch(batch, metrics, llm4scoring, embeddings)
            all_results.update(batch_results)

        result_df = pd.DataFrame.from_dict(all_results, orient='index')
        return result_df, dataset.to_dict()

    def save_results_to_json(self, filepath: str = None, details=True):
        """
        Save evaluation results and optionally the testset to a JSON file.

        Parameters
        ----------
        filepath : str, optional
            Path where the JSON file will be saved, by default None.
        details : bool, optional
            Include the answered testset in the JSON output, by default True.
        """
        if filepath is None:
            filepath = f"results/{self.config.name}.json"

        output_dict = {
            "results": self.results.to_dict(),
            "testset": self.testset_answered if details else None
        }

        with open(filepath, 'w') as json_file:
            json.dump(output_dict, json_file, indent=4)
        
        print(f"Results and testset successfully saved to {filepath}")

    async def evaluate(self, testset: pd.DataFrame):
        """
        Evaluate all configurations and store the results.

        Parameters
        ----------
        testset : pd.DataFrame
            Testset to evaluate.
        """
        config_name = self.config.name
        print(f"Evaluating configuration: {config_name}")
        self.results, self.testset_answered = await self.evaluate_config(self.config, testset)