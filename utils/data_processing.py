import pandas as pd
import numpy as np

def inference_input_processing(df: pd.DataFrame) -> np.array:
    """
    The input of the function is a dataframe with the corresponding columns: x1, x2, ..., x6, z,
    and the output is processed data input, ready to broadcast to model
    :param df:
    :return:
    """
    # both tasks must be performed during inference
    df['mask_t1'], df['mask_t2'] = 1, 1

    # ensure columns order
    df = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'z', 'mask_t1', 'mask_t2']]
    return np.array(df.values.tolist())