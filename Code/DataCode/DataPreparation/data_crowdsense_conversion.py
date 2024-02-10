import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os

# -----------
# Directories
directories = {f'c{i}_covid' : f'/Users/rag004/Documents/PhD/Code/HaphazardInputReview/HaphazardInputs/Data/crowdsense/c{i}_covid.csv' for i in range(1, 9)}

# ----------- Utility functions
def clean_data(df: pd.DataFrame):
    
    # assert df.iloc[:, 0].nunique() == 2, "Number of unique labels should be 2 (binary classification problem)"
    print(f'Samples : {df.shape[0]} |  Features : {df.shape[1]-1} before cleaning')

    # # Find and delete the rows all of whose feature values are NaN
    # nan_rows_indices =df[df.iloc[:, 1:].isna().all(axis=1)].index
    # print(f'Indices of rows with all NaN entries: {nan_rows_indices.values}')
    # if len(nan_rows_indices):
    #     print('Deleting all NaN rows...')
    # df.drop(nan_rows_indices, inplace=True)
    # df.reset_index(drop=True, inplace=True)

    # Find and delete the feature whose values are NaN everywhere
    nan_columns = df.columns[df.isna().all()]
    # Convert the Index object to a list
    nan_columns_list = nan_columns.tolist()
    print("Number of features to be dropped: ", len(nan_columns_list))
    if len(nan_columns_list):
        print('Deleting all NaN features...')
    # Drop columns with all NaN values in place
    df.drop(columns=nan_columns_list, inplace=True)

    print(f'Samples : {df.shape[0]} |  Features : {df.shape[1]-1} after cleaning')

# ----------- 
for data, path in directories.items():
    df = pd.read_csv(path)
    print(f'\nData: {data} | Samples : {df.shape[0]} |  Features : {df.shape[1]-1} | Targets = {df.iloc[:, 0].unique()}')
    del df

# ----------- 
list_of_df = []

for data, path in directories.items():
    print(f'\n\n{data}')
    # load csv data
    df = pd.read_csv(path)

    # if df.iloc[:, 0].nunique() == 2:
    # Clean DataFrame
    clean_data(df)

    # Append in 'list_of_df'
    list_of_df.append(df)

# -----------
# Dropping 3 instances with no observations
num_feat = list_of_df[0].iloc[:,1:].shape[1]
for i in range(len(list_of_df)):
    index_with_no_val = np.where(np.sum(np.isnan(list_of_df[i].iloc[:,1:]), axis = 1) == num_feat)
    list_of_df[i] = list_of_df[i].drop(index_with_no_val[0])

# -----------
# We consider two datasets with binary response: Data: c3_covid and Data: c5_covid

data_c3 = list_of_df[2]
data_c5 = list_of_df[4]

data_c3.loc[data_c3.loc[:,'0'] == 2,'0'] = 1
data_c3[np.isnan(data_c3)] = -1
data_c5[np.isnan(data_c5)] = -1

# ----------- Save Datasets
__file__ = os.path.abspath('')
# Save the dataframe in the .pickle file for easy loading
data_frame_arr_c3 = np.array(data_c3)
data_save_path = os.path.join(os.path.dirname(__file__), 'Data',
                          'crowdsense_c3/crowdsense_c3.pickle')
with open(data_save_path, 'wb') as handle:
    pickle.dump(data_frame_arr_c3, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the dataframe in the .pickle file for easy loading
data_frame_arr_c5 = np.array(data_c5)
data_save_path = os.path.join(os.path.dirname(__file__), 'Data',
                          'crowdsense_c5/crowdsense_c5.pickle')
with open(data_save_path, 'wb') as handle:
    pickle.dump(data_frame_arr_c5, handle, protocol=pickle.HIGHEST_PROTOCOL)
