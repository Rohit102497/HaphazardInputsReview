import pickle
import re
import numpy as np
import pandas as pd
import os

__file__ = os.path.abspath('')
# We consider the train dataset containing 25000 datapoints
data_path = os.path.join(os.path.dirname(__file__), 'Data',
                          'imdb/aclImdb/train/labeledBow.feat')
data = pd.read_csv(data_path, header = None, engine = 'python')
data_arr = np.array(data)
number_of_instances = 25000
# We consider only the first 7500 most common feature
n_feat = 7500
data_frame = pd.DataFrame(-1, index=range(number_of_instances),
                        columns = list(range(n_feat)))

# Each value corresponding to a feature(word) contains the number of times a word 
# was seen in the sentence. If the word was not seen, then maybe the word 
# was not available. Note that, it might be present but we don't know about it.
# We denote it by -1
label = []
for i in range(number_of_instances):
    split = re.findall(r'\d+', data_arr[i][0])
    label.append(int(split[0]))
    index = split[1:][::2]
    value = split[1:][1::2]
    for j in range(len(index)):
        if int(index[j]) < 7500:
            data_frame.loc[i, int(index[j])] = int(value[j])
data_frame.insert(0, column='class', value=label)

# Save the dataframe in the .pickle file for easy loading
data_frame_arr = np.array(data_frame)
data_save_path = os.path.join(os.path.dirname(__file__), 'Data',
                          'imdb/imdb.pickle')
with open(data_save_path, 'wb') as handle:
    pickle.dump(data_frame_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # To load the file
# with open(data_save_path, 'rb') as handle:
#     data_new = pickle.load(handle)