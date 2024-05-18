#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

import numpy as np
import pandas as pd


def pivot_and_stack(df, index_id, column_ids, values_ids) -> np.array:
    """
    Pivot the DataFrame for each feature column and stack the pivoted DataFrames.

    Parameters:
    - dataset: pd.DataFrame, the dataset to pivot.
    - index_col: str, the name of the column to use as the DataFrame index.
    - columns_col: str, the name of the column to use as the DataFrame columns.
    - values_cols: list, a list of column names to pivot and stack.

    Returns:
    - A 3D numpy array with shape [time, tickers, features].
    """
    pivoted_dataset = [df.pivot(index=index_id, columns=column_ids, values=values_id) for values_id in values_ids]
    return np.stack([feature.values for feature in pivoted_dataset], axis=-1)


def generate_batches(data, batch_size):
    """
    Generator function for creating batches of data.

    Parameters:
    - data: 3D numpy array, shape [time, tickers, features]
    - batch_size: int, the size of each batch

    Yields:
    - Batches of data with shape [batch_size, tickers, features]
    """
    num_batches = data.shape[0] // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        yield data[start_idx:end_idx]

    # Handle the last batch if the dataset size isn't a multiple of batch_size
    if data.shape[0] % batch_size != 0:
        yield data[num_batches * batch_size:]
