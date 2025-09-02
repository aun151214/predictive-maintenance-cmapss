import numpy as np

def gen_sequence(df, seq_length, features):
    """
    Generates overlapping sequences of length `seq_length`
    from the dataframe for the given feature columns.
    
    Args:
        df (pd.DataFrame): engine dataframe
        seq_length (int): window length
        features (list): list of feature column names

    Returns:
        Generator of np.ndarray sequences
    """
    data_array = df[features].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


def gen_labels(df, seq_length, label):
    """
    Generates labels corresponding to each sequence
    (the RUL value at the end of each window).
    
    Args:
        df (pd.DataFrame): engine dataframe
        seq_length (int): window length
        label (str): target column (e.g., "RUL")

    Returns:
        np.ndarray: label values
    """
    label_array = df[label].values
    return label_array[seq_length:]
