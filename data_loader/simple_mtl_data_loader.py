import numpy as np
import pandas as pd


def custom_data_loader(df: pd.DataFrame, batchSize: int, data_set_id_column="DatasetID") -> list:
    """
    Takes in a single dataframe containing data sets from both tasks,
    with the assumption that the amount of samples is different in both
    tasks and that the dataset for task 2 has more samples than the dataset of task 1.
    An additional mask variable is created for both tasks

    :param df: dataframe containing data for both tasks
    :param batchSize: batch size from each task dataset (final batch size is 2*batchSize)
    :param data_set_id_column: column indicating the task data
    :return: input batch for model training consisting of a common X (explanatory variables) for
    both tasks and separate y1, y2 (response variables) for the corresponding tasks
    """

    df['mask_t1'] = np.where(np.any(np.isnan(df[['z']]), axis=1), 0, 1)
    df['mask_t2'] = np.where(np.any(np.isnan(df[['z']]), axis=1), 1, 0)

    df = df.replace(np.nan, 0)

    data_1 = df[df[data_set_id_column] == 2]
    data_1 = data_1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'z', 'mask_t1', 'mask_t2', 'y1', 'y2']]

    data_2 = df[df[data_set_id_column] == 1]
    data_2 = data_2[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'z', 'mask_t1', 'mask_t2', 'y1', 'y2']]

    l1 = len(data_1)
    l2 = len(data_2)

    # batches per epoch = steps_per_epoch
    batches = l2 // batchSize
    if l1 % batchSize > 0:
        batches += 1

    while True:
        # shuffle indices in every epoch
        data_1 = data_1.sample(frac=1).reset_index(drop=True)
        data_2 = data_2.sample(frac=1).reset_index(drop=True)

        # batch loop
        for b in range(batches):
            start = b * batchSize
            end = start + batchSize

            # take equal number of samples from each batch
            if (end > data_2.index).all():
                data_2 = data_2.sample(frac=1).reset_index(drop=True)
                data_2.index = range(end - batchSize, end + l2 - batchSize)

            tmp_d1 = data_1.iloc[start:end, :]
            tmp_d2 = data_2.iloc[start:end, :]
            both_datasets = pd.concat([tmp_d1, tmp_d2], ignore_index=True).sample(frac=1).reset_index(drop=True)

            X = np.array(both_datasets.iloc[:, :-2].values.tolist())
            y_t1 = np.array(both_datasets['y1'].values.tolist())
            y_t2 = np.array(both_datasets['y2'].values.tolist())
            yield [X, {'task_1': y_t1, 'task_2': y_t2}]
