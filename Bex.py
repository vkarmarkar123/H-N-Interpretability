import numpy as np
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd

def bex(subset_df):
    def compute_distance(row1, row2):
        """ Compute Pair-wise Euclidean distance """
        return np.sqrt(
            (row1['WLI_x_coord'] - row2['WLI_x_coord']) ** 2 + (row1['WLI_y_coord'] - row2['WLI_y_coord']) ** 2)

    def process_data(df):

        if set(df['Label'].unique()) == {0, 1}:
            # Create a DataFrame for each label
            df_label_0 = df[df['Label'] == 0]
            df_label_1 = df[df['Label'] == 1]

            # Calculate distances and filter out rows with distance < 20 pixels
            to_drop = []
            for index_0, row_0 in df_label_0.iterrows():
                for index_1, row_1 in df_label_1.iterrows():
                    if compute_distance(row_0, row_1) < 20:
                        to_drop.extend([index_0, index_1])

            df = df.drop(to_drop).reset_index(drop=True)

        return df

    caseID = subset_df['Case']
    cases = np.unique(caseID)
    train_batch = []

    for Case in cases:
        case_batch = subset_df[subset_df['Case'] == Case]
        Run_ID = case_batch['Run']
        Runs = np.unique(Run_ID)

        for Run in Runs:
            run_batch = case_batch[case_batch['Run'] == Run]
            processed_df = process_data(run_batch)

    return processed_df


#Bex.py achieves a similar goal as Fast_BEX, but does it differently

# - The iterating over 'Case' and 'Run' is similar, along with the separation of data
#   based on 0s and 1s
# - It calculates Euclidean distance using a nested loop and a custom function (compares every point labeled 0 with every point labeled a 1)
# - if dist < 20, both points marked for removal
