import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def optimized_bex(subset_df):
    def process_data(df):
        if set(df['Label'].unique()) == {0, 1}:
            # Splitting the DataFrame based on Label
            df_label_0 = df[df['Label'] == 0]
            df_label_1 = df[df['Label'] == 1]

            # Compute pairwise distances using cdist from scipy
            distances = cdist(df_label_0[['WLI_x_coord', 'WLI_y_coord']], df_label_1[['WLI_x_coord', 'WLI_y_coord']], metric='euclidean')
            close_pairs = np.where(distances < 20)

            # Indexes to drop
            to_drop_0 = df_label_0.iloc[close_pairs[0]].index
            to_drop_1 = df_label_1.iloc[close_pairs[1]].index
            to_drop = to_drop_0.union(to_drop_1)

            df = df.drop(to_drop).reset_index(drop=True)
        return df

    processed_dfs = []
    for Case, case_group in subset_df.groupby('Case'):
        for Run, run_group in case_group.groupby('Run'):
            processed_dfs.append(process_data(run_group))

    return pd.concat(processed_dfs, ignore_index=True)


# optimized_bex() processes a dataframe by:

# - grouping the data by 'Case' and within each 'Case' by 'Run'
# - separates data into 2 subsets based on 'Label' value (0,1)
# - Calculates pairwise Euclidean Distance between all points in the 2 subsets (using WLI_x_coord and WLI_y_coord)
# - Identifies pairs of points (one from each label group) that are closer than 20 units apart
# - removes said points
# - concatenates all processed groups to a return DF