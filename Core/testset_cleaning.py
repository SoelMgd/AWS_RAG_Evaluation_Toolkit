import re
import pandas as pd
from typing import Union

def clean_question(text: str) -> str:
    """
    Cleans a question by removing introductory phrases such as 
    'Based on the context, here is a question:'.

    Args:
        text (str): The input text containing the question.

    Returns:
        str: The cleaned question text.
    """
    pattern = r'(?:.*?:\s*)?(.*?\?)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def remove_leading_quote(text: str) -> str:
    """
    Removes leading quotation marks from the beginning of a string.

    Args:
        text (str): The input string with possible leading quotes.

    Returns:
        str: The string with leading quotes removed.
    """
    return re.sub(r'^"', '', text)

def remove_prefixes(text: str) -> str:
    """
    Removes predefined prefixes from the beginning of a text.

    Args:
        text (str): The input text that may contain prefixes.

    Returns:
        str: The text with prefixes removed.
    """
    prefixes = [
        r"According to the given context,\s*",
        r"According to the context,\s*",
        r"Based on the given context,\s*",
        r"Based on the context,\s*",
        r"The context advises\s*"
    ]
    
    pattern = r"^(" + "|".join(prefixes) + ")"
    
    cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    if cleaned_text:
        cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
    
    return cleaned_text

def starts_with_no_context(text: str) -> bool:
    """
    Determines whether the text indicates that the context does not allow answering the question.

    Args:
        text (str): The input text to be checked.

    Returns:
        bool: True if the text indicates insufficient context, otherwise False.
    """
    start_patterns = r"^\s*(The (given )?context does not)"
    insufficient_info_pattern = r"does not provide enough information to answer the question"
    
    starts_with_no_context = bool(re.match(start_patterns, text, re.IGNORECASE))
    contains_insufficient_info = bool(re.search(insufficient_info_pattern, text, re.IGNORECASE))
    
    return starts_with_no_context or contains_insufficient_info

def clean_testset(testset: pd.DataFrame) -> pd.DataFrame:
    """
    Applies pre-processing functions to a DataFrame containing questions and ground truth.

    Args:
        testset (pd.DataFrame): The DataFrame with columns 'question' and 'ground_truth'.

    Returns:
        pd.DataFrame: The DataFrame with pre-processed 'question' and 'ground_truth' columns, and a new 'not legit' column.
    """
    testset['question'] = testset['question'].apply(clean_question)
    testset['question'] = testset['question'].apply(remove_prefixes)
    testset['question'] = testset['question'].apply(remove_leading_quote)
    testset['ground_truth'] = testset['ground_truth'].apply(remove_prefixes)
    testset['not_legit'] = testset['ground_truth'].apply(starts_with_no_context)
    
    return testset

def display_row_info(testset: pd.DataFrame, row_number: int) -> None:
    """
    Displays specific information from a row in the testset DataFrame, including question, context, and ground truth.

    Args:
        testset (pd.DataFrame): The DataFrame containing the data.
        row_number (int): The index of the row to display.

    Returns:
        None
    """
    if row_number < 0 or row_number >= len(testset):
        print(f"Error: Row number {row_number} is out of bounds.")
        return
    
    row = testset.iloc[row_number]
    
    print(f"Information for row {row_number}:")
    print(f"\nQuestion:\n{row['question']}")
    print(f"\nRAGAS Context:\n{row['contexts']}")
    print(f"\nFile used by RAGAS:\n{row['metadata']}")
    print(f"\nRAGAS Ground Truth:\n{row['ground_truth']}")