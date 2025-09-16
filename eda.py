'''
Oppotunities for more eda
- look for correlations with stopwords
- fix sentiment issues or correlate to ratings in another way
- explore drug and conditions vs rating
'''

from datasets import load_dataset
import pandas as pd
from typing import Optional, Union, List
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def read_hf_dataset_columns(
    dataset_name: str,
    columns: List[str] = ["text", "length", "rating"],
    split: str = "train",
    return_format: str = "pandas"
) -> Union[pd.DataFrame, dict]:
    """
    Read a Hugging Face dataset and extract specific columns.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub
        columns (List[str]): List of column names to extract
        return_format (str): Return format - "pandas" or "dict" (default: "pandas")
    
    Returns:
        pd.DataFrame or dict: Dataset with specified columns
    
    Raises:
        ValueError: If specified columns don't exist in the dataset
        Exception: If dataset loading fails
    
    Note: Claude
    """
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split=split)
        
        # Check if all requested columns exist
        available_columns = dataset.column_names
        missing_columns = [col for col in columns if col not in available_columns]
        
        if missing_columns:
            print(f"Warning: Columns {missing_columns} not found in dataset.")
            print(f"Available columns: {available_columns}")
            # Filter to only existing columns
            columns = [col for col in columns if col in available_columns]
        
        if not columns:
            raise ValueError("None of the specified columns exist in the dataset")
        
        # Extract the specified columns
        extracted_data = dataset.select_columns(columns)
        
        # Return in requested format
        if return_format.lower() == "pandas":
            return extracted_data.to_pandas()
        else:
            return extracted_data.to_dict()
            
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {str(e)}")
        raise e

def get_plots_of_lengths(df):
    """
    make a series of plots to examine the lengths of the reviews
    
    Args:
        df (dataframe): dataframe to get plot made of
        
    Result:
       show plot on screens
    """

    plt.hist(df['rating'], bins=10, label='ratings', color='blue')
    plt.show()
    plt.hist(df['review_length'], bins=50, label='lengths', color='blue')
    plt.show()
    
    plt.hist(df[df["rating"] < 3]['review_length'], bins=50, alpha=0.5, label='low', color='blue')
    plt.hist(df[df["rating"] >= 7]['review_length'], bins=50, alpha=0.5, label='high', color='red')
    plt.show()

    plt.hist(df[df["rating"] < 3]['symbol_count'], bins=50, alpha=0.5, label='low', color='blue')
    plt.hist(df[df["rating"] >= 7]['symbol_count'], bins=50, alpha=0.5, label='high', color='red')
    plt.show()

    plt.hist(df[df["rating"] < 3]['letter_count'], bins=50, alpha=0.5, label='low', color='blue')
    plt.hist(df[df["rating"] >= 7]['letter_count'], bins=50, alpha=0.5, label='high', color='red')
    plt.show()
    
    df.plot.scatter(x= "review_length", y="rating", logx=True)
    plt.show()

def count_letters(text):
    count = 0
    for char in text:
        if char in "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM":
            count += 1
    return count

def count_sym(text):
    count = 0
    for char in text:
        if char in "!@#$%^&*(){}[]_+-=:;\'\'\\|,./<>?":
            count += 1
    return count

def get_counts_of_characters(df):
    """
    count characters of different types of characters
    
    Args:
        df (dataframe): dataframe to get plot made of
        
    Result:
       show plot on screens
    """

    df['letter_count'] = df['review'].apply(count_letters)
    df['symbol_count'] = df['review'].apply(count_sym)
    print("Added word count:")
    print(df[['review', 'letter_count', 'symbol_count']].head())

def calculate_sentiment(df):
    '''
    Use the nltk to get sentiment analysis
    Unfortunately pandas is not working as expected.
    '''

    # Initialize the VADER sentiment intensity analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Get polarity scores
    sentiment_scores = df['review'].apply(analyzer.polarity_scores).to_dict()
    print(sentiment_scores[0].keys())
    sent_df=pd.DataFrame(sentiment_scores.values(),index=sentiment_scores.keys())
    print(sent_df.head())
    df = pd.concat([df, sent_df])
    plt.hist(df["pos"], bins=50, label='pos', alpha=0.9, color='blue')
    plt.hist(df["neu"], bins=50, label='neu', alpha=0.5, color='yellow')
    plt.hist(df["neg"], bins=50, label='neg', alpha=0.5, color='red')
    plt.show()

    plt.hist(df[df["rating"] < 3]['pos'], bins=50, alpha=0.5, label='low', color='blue')
    plt.hist(df[df["rating"] >= 7]['pos'], bins=50, alpha=0.5, label='high', color='red')
    plt.show()

    plt.hist(df[df["rating"] < 3]['neu'], bins=50, alpha=0.5, label='low', color='blue')
    plt.hist(df[df["rating"] >= 7]['neu'], bins=50, alpha=0.5, label='high', color='red')
    plt.show()

    plt.hist(df[df["rating"] < 3]['neg'], bins=50, alpha=0.5, label='low', color='blue')
    plt.hist(df[df["rating"] >= 7]['neg'], bins=50, alpha=0.5, label='high', color='red')
    plt.show()

    print(df.head())

# Example usage
if __name__ == "__main__":
    
    drug_review_dataset = "flxclxc/encoded_drug_reviews"
    # Columns: patient_id drugName condition review rating date usefulCount review_length encoded

    # Replace with your actual dataset name
    df = read_hf_dataset_columns(
        dataset_name=drug_review_dataset,
        columns=["review", "rating", "usefulCount", "review_length"],
        split='train'
    )
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
        
    # get_counts_of_characters(df)
    # get_plots_of_lengths(df)
    calculate_sentiment(df)

