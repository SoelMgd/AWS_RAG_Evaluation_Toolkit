import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ResultsAnalyzer:
    """
    Class to analyze and visualize results from various models on specific metrics.

    Attributes:
    ----------
    results : dict
        Stores the results from different JSON files.
    testsets : dict
        Stores the testsets corresponding to the results.
    """

    def __init__(self, json_files):
        """
        Initialize the ResultsAnalyzer with a list of JSON file paths.

        Parameters:
        ----------
        json_files : list
            A list of file paths to JSON files containing the results.
        """
        self.results = {}
        self.testsets = {}
        self.load_results(json_files)

    def load_results(self, json_files):
        """
        Load results and optional testsets from JSON files into dictionaries.

        Parameters:
        ----------
        json_files : list
            A list of file paths to JSON files containing the results.
        """
        for file in json_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"The file {file} does not exist.")
            
            with open(file, 'r') as f:
                data = json.load(f)
                
                file_name = os.path.splitext(os.path.basename(file))[0]
                
                # Load and convert results to DataFrames
                results = data.get("results", {})
                self.results[file_name] = {metric: pd.Series(scores) for metric, scores in results.items()}
                
                # Load testset if available
                self.testsets[file_name] = data.get("testset", None)

    def show_heatmap(self, metric, file_names=None):
        """
        Plot a heatmap for a specific metric and selected file names.
        Rows are sorted by the average score for the given metric, but the questions
        on the x-axis remain in their original order.

        Parameters:
        ----------
        metric : str
            The metric to plot the heatmap for.
        file_names : list, optional
            List of file names to include in the heatmap. If None, include all.
        """
        if file_names is None:
            file_names = list(self.results.keys())

        # Extract data for the heatmap
        heatmap_data = {name: metrics[metric].values for name, metrics in self.results.items()
                        if name in file_names and metric in metrics}

        if not heatmap_data:
            print(f"No data available for metric '{metric}' with selected file names.")
            return

        # Convert data to a DataFrame
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Add 'average' column to compute the average score per model
        heatmap_df['average'] = heatmap_df.mean(axis=1)
        
        # Sort configurations by average score (rows), keep questions in original order (columns)
        heatmap_df_sorted = heatmap_df.sort_values(by='average', ascending=False).drop(columns='average')

        # Plot heatmap
        plt.figure(figsize=(20, 5))
        sns.heatmap(heatmap_df_sorted.T, cmap='YlGnBu', cbar=True, annot=True, fmt=".2f")
        
        # Set plot titles and labels
        plt.title(f"{metric} Heatmap")
        plt.xlabel("Questions")
        plt.ylabel("Configurations")
        
        # Ensure x-axis (questions) are in the original order
        plt.xticks(ticks=range(heatmap_df_sorted.shape[0]), labels=range(heatmap_df_sorted.shape[0]), rotation=45)
        plt.yticks(rotation=0)
        plt.show()

    def show_all_heatmaps(self, file_names=None):
        """
        Plot heatmaps for all metrics present in the results for selected file names.

        Parameters:
        ----------
        file_names : list, optional
            List of file names to include in the heatmaps. If None, include all.
        """
        if file_names is None:
            file_names = list(self.results.keys())

        # Collect all unique metrics
        metrics = set()
        for file_name in file_names:
            if file_name in self.results:
                metrics.update(self.results[file_name].keys())
        
        # Generate heatmaps for each metric
        for metric in metrics:
            self.show_heatmap(metric, file_names)

    def report(self, threshold_dict=None, file_names=None):
        """
        Report overall statistics for each configuration and metric, including the ratio of 
        questions exceeding specific thresholds for specified metrics.

        Parameters:
        ----------
        threshold_dict : dict, optional
            A dictionary where keys are metric names and values are thresholds. Example:
            {"faithfulness": 0.9, "context_recall": 0.7}
        file_names : list, optional
            List of file names to include in the report. If None, include all.
        """
        if threshold_dict is None:
            threshold_dict = {}
        if file_names is None:
            file_names = list(self.results.keys())

        for file_name in file_names:
            if file_name not in self.results:
                print(f"No data available for result name '{file_name}'.")
                continue

            metrics_dict = self.results[file_name]
            print(20 * '=', file_name, (30 - len(file_name)) * '=')

            for metric, series in metrics_dict.items():
                average_score = series.mean()
                print(f"Average {metric}: {average_score:.2f}")

                # Check if threshold exists for this metric
                if metric in threshold_dict:
                    threshold = threshold_dict[metric]
                    above_threshold_ratio = (series > threshold).mean()
                    num_questions_above = (series > threshold).sum()
                    print(f"{num_questions_above} questions ({above_threshold_ratio * 100:.2f}%) "
                          f"exceeded the threshold of {threshold} for {metric}")

            print("\n")

    def barplot(self, metrics=None, file_names=None):
        """
        Plot a barplot for each specified metric and configuration. Values are displayed on top of the bars.

        Parameters:
        ----------
        metrics : list, optional
            List of metrics to include in the barplot. If None, include all metrics.
        file_names : list, optional
            List of file names to include in the barplot. If None, include all.
        """
        if metrics is None:
            metrics = list({metric for metrics_dict in self.results.values() for metric in metrics_dict.keys()})

        if file_names is None:
            file_names = list(self.results.keys())

        # Prepare data for the barplot
        data = []
        for file_name in file_names:
            if file_name not in self.results:
                print(f"No data available for result name '{file_name}'.")
                continue

            metrics_dict = self.results[file_name]
            for metric, series in metrics_dict.items():
                if metric in metrics:
                    mean_score = series.mean()
                    data.append({'File': file_name, 'Metric': metric, 'Average Score': mean_score})

        df_plot = pd.DataFrame(data)
        metrics = df_plot['Metric'].unique()

        # Plot barplot for each metric
        for metric in metrics:
            plt.figure(figsize=(10,5))
            
            ax = sns.barplot(x='File', y='Average Score', data=df_plot[df_plot['Metric'] == metric], palette='viridis')

            # Annotate each bar with the average score
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

            plt.title(f'Average Scores for Metric: {metric}')
            plt.xlabel('File')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
            plt.show()

    def print_question(self, i, result_name):
        """
        Print details of the question at index i for the given result_name.

        Parameters:
        ----------
        i : int
            The index of the question to print.
        result_name : str
            The name of the result set to use.
        """
        if result_name not in self.testsets:
            print(f"No testset data available for result name '{result_name}'.")
            return

        testset = self.testsets[result_name]
        if testset is None:
            print("Testset is empty.")
            return

        try:
            question = testset['question'][i]
            ground_truth = testset['ground_truth'][i]
            contexts = testset['contexts'][i]
            answer = testset['answer'][i]

            print("="*100)
            print(f"Question {i}\n")
            print(question, '\n')
            print(f"Ground Truth RAGAS")
            print(ground_truth, '\n')
            print(f"Context of the RAG")
            print(contexts, '\n')
            print(f"Answer of the RAG")
            print(answer)
            print("="*100)

        except IndexError:
            print(f"Index {i} is out of bounds for the testset '{result_name}'.")
