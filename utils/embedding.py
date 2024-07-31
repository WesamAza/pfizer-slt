import os
import csv
import numpy as np
import pandas as pd

def precompute_embeddings(text_col_name: str, embedding_function: callable, output_file_path : str, pickle_file_path: str = None, data_df : pd.DataFrame = None, ):
    # Load the text data from the pickle file
    if data_df is not None:
        text = data_df[text_col_name]
    elif pickle_file_path is not None:
        text = pd.read_pickle(pickle_file_path)[text_col_name]
    else:
        raise ValueError('Either data_df or pickle_file_path must be provided')

    # Generate embeddings using the provided embedding function
    embeddings = embedding_function(text)

    # Save the embeddings to the new file path
    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(embeddings)
    
    # Return the path of the embeddings file
    return np.array(embeddings, dtype=float)

def read_embeddings(embeddings_file_path: str):
    # Load the embeddings from the file
    with open(embeddings_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        embeddings = list(reader)
    
    # Return the embeddings
    return np.array(embeddings, dtype=float)