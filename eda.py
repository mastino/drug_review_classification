from datasets import load_dataset
import pandas as pd
from typing import Optional, Union, List
import matplotlib.pyplot as plt


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
        raise

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
        
    hist1 = df['review_length'].hist(bins=30)
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()
        
    hist2 = df[['length', 'rating']].hist(bins=25, figsize=(12, 5))
    plt.suptitle('Distribution of Length and Rating')
    plt.tight_layout()
    plt.show()