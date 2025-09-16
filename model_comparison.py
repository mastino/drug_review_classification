import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Optional, Union, List
from datasets import load_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

'''
Things to compare
- Try word embeddings
- include length, sentiment, useful count as features

'''



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
    
    Claude
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

def process_review(review):
    '''
    convert a single review to str tokens

    Args:
      review: string with review

    Returns:
      tokens: string of tokens

    Claude with some modification
    '''
    review = ' '.join(review.split())
    
    tokens = word_tokenize(review)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def process_data(df):
    '''
    convert dataframe to input embeddings and output

    Args:
      df: dataframe with the columns "review", "rating", "usefulCount", "review_length"

    Returns
      X: input features for training
      y: rating of the review

    Some from Claude
    '''

    df = df.dropna(subset=["rating"])
    df["review"] = df['review'].fillna('')
    df['review_tokens'] = df['review'].apply(process_review)
  
    # TODO vary max features
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )

    X = vectorizer.fit_transform(df["review_tokens"])
    y = df["rating"].to_numpy()  # or convert ratings to binary

    return X,y

def compare_models(X_train, X_test, y_train, y_test):
    '''
    Compare logistic regression, random forest, and SVM
    Args:
        X_train, X_test:
        y_train, y_test:
    Results:
        output model accuracies to stdout

    Claude suggested the dictionary and the classification_report
    '''

    print("model init....")
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf')
    }

    for name, model in models.items():
        print(f"training and testing {name}....")
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()
        accuracy = accuracy_score(y_test, predictions)
        train_time = fit_time - start_time
        test_time = end_time - fit_time

        print(f"{name}: {accuracy:.4f}")
        print(f"  train time: {train_time:.4f}s")
        print(f"  test time:  {test_time:.4f}s")
        print(classification_report(y_test, predictions))


def plot_grid(param_1, param_2, result):
    '''
    Plot a color gradient to express result relative to the params
    Args:
        param_1 - parameter shown on x axis
        param_2 - 
        result

    Assist from Google cuase I searched for the matplot lib function
      and it just gave the answer ¯\\_(ツ)_/¯ 
    '''
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(param_1, param_2, result, cmap='plasma')
    plt.colorbar(mesh, ax=ax, label='Accuracy')
    plt.show()

def sweep_hyperparameters(X_train, X_test, y_train, y_test):
    '''
    Compare logistic regression, random forest, and SVM
    Args:
        X_train, X_test:
        y_train, y_test:
    Results:
        output model accuracies to stdout
    '''

    print("model init....")
    # TODO make optinos for the other models
    # for name, model in models.items():
    n_estimators_li = np.arange(75,300,25)
    min_samples_li = np.arange(2,21,2)
    accuracy_li = np.ones(n_estimators_li.size*min_samples_li.size)

    acc_count = 0
    for n_est in n_estimators_li:
        for min_samp in min_samples_li:
            model = RandomForestClassifier(n_estimators=n_est, min_samples_split=min_samp, random_state=42)

            print(f"training and testing n estimators={n_est} min_samples={min_samp}....")
            start_time = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time()
            predictions = model.predict(X_test)
            end_time = time.time()
            accuracy = accuracy_score(y_test, predictions)
            train_time = fit_time - start_time
            test_time = end_time - fit_time

            print(f"n estimators={n_est} min_samples={min_samp}")
            print(f"  accuracy:   {accuracy:.4f}")
            print(f"  train time: {train_time:.4f}s")
            print(f"  test time:  {test_time:.4f}s")
            report = classification_report(y_test, predictions, output_dict=True)
            accuracy_li[acc_count] = report['accuracy']

            print(report['accuracy'])
            print()

    accuracy_li = accuracy_li.reshape(n_estimators_li.size,min_samples_li.size)
    for row in accuracy_li:
        for acc in row:
            print(f"{acc:5.2f}", end=' ')
        print()
    # plot_grid(n_estimators_line, min_samples_line, accuracy_li)

def run_model():
    n_est = 100
    min_samp = 5
    model = RandomForestClassifier(n_estimators=n_est, min_samples_split=min_samp, random_state=42)

    print(f"training and testing n estimators={n_est} min_samples={min_samp}....")
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time()
    predictions = model.predict(X_test)
    end_time = time.time()
    accuracy = accuracy_score(y_test, predictions)
    train_time = fit_time - start_time
    test_time = end_time - fit_time

    print(f"n estimators={n_est} min_samples={min_samp}")
    print(f"  accuracy:   {accuracy:.4f}")
    print(f"  train time: {train_time:.4f}s")
    print(f"  test time:  {test_time:.4f}s")
    print(classification_report(y_test, predictions, output_dict=True))


if __name__ == "__main__":
    
    # argparse courtesy of Caude (mostly)
    # Simple argument parser for model operations
    parser = argparse.ArgumentParser(description='Select model operation to run')
    parser.add_argument('-o', '--operation', choices=['compare_models', 'sweep_hyperparameters', 'run_model'], 
                        required=True, help='Choose operation: compare_models, sweep_hyperparameters, or run_model')
    args = parser.parse_args()

    # Or with the variable approach:
    # check out this approach the LLM suggested!
    # operations = {
    #     'compare_models': lambda: print("Comparing multiple models..."),
    #     'sweep_hyperparameters': lambda: print("Sweeping hyperparameters..."),
    #     'run_model': lambda: print("Running model training...")
    # }
    # operations[operation]()

    drug_review_dataset = "flxclxc/encoded_drug_reviews"
    # Columns: patient_id drugName condition review rating date usefulCount review_length encoded

    print("loading data....")
    start_time = time.time()
    # note this dataset only has train
    df = read_hf_dataset_columns(
        dataset_name=drug_review_dataset,
        columns=["review", "rating", "usefulCount", "review_length"],
        split='train'
    )
    end_time = time.time() - start_time
    print(f"{end_time:.4f}s")

    print("process data....")
    start_time = time.time()
    X,y = process_data(df)
    end_time = time.time() - start_time
    print(f"{end_time:.4f}s")
    print("make train test split....")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    end_time = time.time() - start_time
    print(f"{end_time:.4f}s")

    if args.operation == 'compare_models':
        compare_models(X_train, X_test, y_train, y_test)
    elif args.operation == 'sweep_hyperparameters':
        sweep_hyperparameters(X_train, X_test, y_train, y_test)
    elif args.operation == 'run_model':
        run_model(X_train, X_test, y_train, y_test)
    else:
        print("oops I don't know what you want")

        