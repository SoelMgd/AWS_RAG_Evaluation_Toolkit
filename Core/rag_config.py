class RAGConfig:
    """
    Configuration class for Retrieval-Augmented Generation (RAG).

    Attributes:
        name (str): The name of the configuration.
        generation_llm (str): The model name for generation.
        embedder (str): The model name for embedding.
        prompt (str): The prompt used for the query.
        scoring_llm (str): The model name for scoring.
        metrics (list): List of metrics to evaluate the responses.
    """

    def __init__(self, name, generation_llm, embedder, prompt, scoring_llm, metrics):
        """
        Initialize the RAGConfig with provided parameters.

        Args:
            name (str): The name of the configuration.
            generation_llm (str): The model name for generation.
            embedder (str): The model name for embedding.
            prompt (str): The prompt used for the query.
            scoring_llm (str): The model name for scoring.
            metrics (list): List of metrics to evaluate the responses.
        """
        self.name = name
        self.generation_llm = generation_llm
        self.embedder = embedder
        self.prompt = prompt
        self.scoring_llm = scoring_llm
        self.metrics = metrics

    def __repr__(self):
        """
        Return a string representation of the RAGConfig instance.

        Returns:
            str: A string that includes the configuration details.
        """
        return (f"RAGConfig(name={self.name}, generation_llm={self.generation_llm}, "
                f"embedder={self.embedder}, metrics={self.metrics}, "
                f"scoring_llm={self.scoring_llm}, prompt_length={len(self.prompt)})")
