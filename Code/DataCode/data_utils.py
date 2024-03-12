# Libraries required
import numpy as np
import pandas as pd
import os
import pickle
# import sys

def data_folder_path(data_folder, data_name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data', data_folder, data_name)

# Load wpbc data
def data_load_wpbc(data_folder):
    data_name = "wpbc.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[1] == "R")*1
    data_initial = data_initial.iloc[:,2:]
    set_nan_index = np.where(np.array(data_initial[34]) == '?')
    data_initial.loc[set_nan_index[0], 34] = np.nan
    data_initial[34] = np.array(data_initial[34]).astype(float)
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load ionosphere data
def data_load_ionosphere(data_folder):
    data_name = "ionosphere.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[34] == "g")*1
    data_initial = data_initial.iloc[:,:34]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y


# Load wdbc data
def data_load_wdbc(data_folder):
    data_name = "wdbc.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[1] == "M")*1
    data_initial = data_initial.iloc[:,2:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y


# Load australian data
def data_load_australian(data_folder):
    data_name = "australian.dat"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = " " , header = None, engine = 'python')
    label = np.array(data_initial[14] == 1)*1
    data_initial = data_initial.iloc[:,:14]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load credit_a data
def data_load_credit_a(data_folder):
    data_name = "credit_a.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[15] == '+')*1
    data_initial = data_initial.iloc[:,:15]
    set_nan_index = np.where(data_initial == '?')
    # features which are categorical
    feat = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    for i in feat:
        val_list = list(set(np.unique(data_initial[i])).difference({'?'}))
        for j in range(len(val_list)):
            data_initial.loc[data_initial[i] == val_list[j], i] = j+1
    for i in range(len(set_nan_index[0])):
        data_initial.loc[set_nan_index[0][i], set_nan_index[1][i]] = np.nan
    for i in range(data_initial.shape[1]):
        data_initial[i] = np.array(data_initial[i]).astype(float)
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y


# Load wbc data
def data_load_wbc(data_folder):
    data_name = "wbc.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[10] == 4)*1
    data_initial = data_initial.iloc[:,1:10]
    set_nan_index = np.where(np.array(data_initial[6]) == '?')
    data_initial.loc[set_nan_index[0], 6] = np.nan
    # data_initial[6][set_nan_index[0]] = np.nan
    data_initial[6] = np.array(data_initial[6]).astype(float)
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load diabetes data
def data_load_diabetes_f(data_folder):
    data_name = "diabetes_f.csv"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, header = None, engine = 'python')
    data_initial = data_initial[1:]
    for i in range(data_initial.shape[1]):
        data_initial[i] = data_initial[i].astype(float)
    label = np.array(data_initial[8] == 1)*1
    data_initial = data_initial.iloc[:, :8]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load german data
def data_load_german(data_folder):
    data_name = "german.data-numeric"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "  " , header = None, engine = 'python')
    # Remove 41 instances which have label as "nan"
    data_initial = data_initial.iloc[np.where(data_initial[24].isnull() == False)[0],:]
    label = np.array(data_initial[24] == 1)*1
    data_initial = data_initial.iloc[:,:24]
    data_initial.insert(0, column="class", value=label)
    for i in range(data_initial.shape[0]):
        data_initial.iloc[i,3] = int(data_initial.iloc[i,3].split(" ")[1])
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1], dtype = float)
    X = np.array(data.iloc[:,1:], dtype = float)

    return X, Y

# Load Italy Power Demand data
def data_load_ipd(data_folder):
    data_name_train = "ItalyPowerDemand_TRAIN.txt"
    data_name_test  = "ItalyPowerDemand_TEST.txt"
    data_path_train = data_folder_path(data_folder, data_name_train)
    data_path_test = data_folder_path(data_folder, data_name_test)
    data_initial_train =  pd.read_csv(data_path_train, sep = "  " , header = None, engine = 'python')
    data_initial_test =  pd.read_csv(data_path_test, sep = "  " , header = None, engine = 'python')
    data_initial = pd.concat([data_initial_train, data_initial_test])

    label = np.array(data_initial[0] == 1)*1
    data_initial = data_initial.iloc[:,1:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)
    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load svmguide3 data
def data_load_svmguide3(data_folder):
    data_name = "svmguide3.txt"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep=" ", header=None, engine = 'python')
    data_initial = data_initial.iloc[:, :22]
    for j in range(1, data_initial.shape[1]):
            for i in range(data_initial.shape[0]):
                    data_initial.iloc[i, j] = data_initial.iloc[i, j].split(":")[1]
    for i in range(data_initial.shape[1]):
        data_initial[i] = data_initial[i].astype(float)

    label = np.array(data_initial[0] == -1)*1
    data_initial = data_initial.iloc[:,1:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load krvskp data
def data_load_krvskp(data_folder):
    data_name = "krvskp.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[36] == 'won')*1
    data_initial = data_initial.iloc[:,:36]
    # The different values of the 36 features are - 'b', 'f', 'g', 'l', 'n', 't', 'w'. 
    # Accordingly we assign them the value - 1, 2, 3, 4, 5, 6, 7 resepectively.
    data_initial[data_initial == 'b'] = 1
    data_initial[data_initial == 'f'] = 2
    data_initial[data_initial == 'g'] = 3
    data_initial[data_initial == 'l'] = 4
    data_initial[data_initial == 'n'] = 5
    data_initial[data_initial == 't'] = 6
    data_initial[data_initial == 'w'] = 7
    for i in range(data_initial.shape[1] - 1):
            data_initial[i] = data_initial[i].astype(float)
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1], dtype = float)
    X = np.array(data.iloc[:,1:], dtype = float)

    return X, Y

# Load spambase data
def data_load_spambase(data_folder):
    data_name = "spambase.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[57] == 1)*1
    data_initial = data_initial.iloc[:,:57]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load magic04 data
def data_load_magic04(data_folder):
    data_name = "magic04.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[10] == 'g')*1
    data_initial = data_initial.iloc[:,:10]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load a8a data
def data_load_a8a(data_folder):
    data_name = "a8a.txt"
    n_feat = 123
    number_of_instances = 32561
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = " ", header = None, engine = 'python')
    data = pd.DataFrame(0, index=range(number_of_instances), columns = list(range(1, n_feat+1)))
    # 16th column contains only NaN value
    data_initial = data_initial.iloc[:, :15]
    for j in range(data_initial.shape[0]):
            l = [int(i.split(":")[0])-1 for i in list(data_initial.iloc[j, 1:]) if not pd.isnull(i)]
            data.iloc[j, l] = 1
    label = np.array(data_initial[0] == -1)*1
    data.insert(0, column='class', value=label)
    data = data.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load susy data
def data_load_susy(data_folder):
    data_name = "SUSY_1M.csv.gz"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, compression='gzip')
    label = np.array(data_initial["0"] == 1.0)*1
    data_initial = data_initial.iloc[:,1:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load susy data
def data_load_higgs(data_folder):
    data_name = "HIGGS_1M.csv.gz"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, compression='gzip')
    label = np.array(data_initial["0"] == 1.0)*1
    data_initial = data_initial.iloc[:,1:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load diabetes us dataset:
def data_load_diabetes_us(data_folder):
    data_name = "diabetes_us.csv"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, header = None, engine = 'python')
    # 1st row contains the header name so we remove that
    data_initial = data_initial[1:]
    # Only readmission days <30 is positive class, otherwise it is a negative class
    label = np.array(data_initial[49] == '<30')*1
    # The last column contains the labels
    data_initial = data_initial.iloc[:,:49]
    # The first two column is admission and patient id, so we drop these
    data_initial = data_initial.iloc[:,2:]
    # "?", "nan" for glucose serum test result (feat no. 22) and "nan" for A1c test result (feat no. 23), is considered
    # unavailable/unmeasured features. For the time being we denote this by "-1" and later we replace it with np.nan
    data_initial[data_initial == '?'] = "-1"
    data_initial[22] = data_initial[22].fillna("-1")
    data_initial[23] = data_initial[23].fillna("-1")
    # The age feature (feat no. 4) indicates only if the age is in a bracket of 10 ([0,10], [10,20], ...). We consider 
    # the median value of the bracket as the actual value and replace it with that
    val_list = sorted(list(set(np.unique(data_initial[4])).difference({"-1"})))
    age_list = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    for j in range(len(val_list)):
        data_initial.loc[data_initial[4] == val_list[j], 4] = age_list[j]
    # Similar to age feature, the weight feature (feat no. 5) is represented in brackets. We consider the median value 
    # and replace with that
    val_list = ['[0-25)', '[25-50)', '[50-75)', '[75-100)', '[100-125)', '[125-150)', '[150-175)', '[175-200)', '>200']
    weight_list = [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5]
    for j in range(len(val_list)):
        data_initial.loc[data_initial[5] == val_list[j], 5] = age_list[j]
    # The below feat_list features contains categorical value. We transform them to numerical value by assigning them
    # value from 1 to the number of categories in each feature
    feat_list = [2, 3, 10, 11, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    for i in feat_list:
        val_list = sorted(list(set(np.unique(data_initial[i])).difference({"-1"})))
        # print(val_list)
        for j in range(len(val_list)):
            data_initial.loc[data_initial[i] == val_list[j], i] = j+1
    # Substitute each position containing -1 with nan value
    data_initial[data_initial == "-1"] = np.nan
    # Convert everything to float type
    for i in data_initial.columns:
            data_initial[i] = np.array(data_initial[i]).astype(float)
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    # Missing columns are - 2,  5, 10, 11, 18, 19, 20, 22, 23 in the orginal dataset (zero indexing).
    # In the processed dataset missing columns are - [ 0,  3,  8,  9, 16, 17, 18, 20, 21] (zero index).

    return X, Y

# Load imdb dataset:
def data_load_imdb(data_folder):
    data_name = "imdb"
    data_path = data_folder_path(data_folder, data_name)
    # To load the file
    with open(data_path, 'rb') as handle:
        data_initial = pickle.load(handle)

    data_initial = data_initial.astype(float)
    # Substitute each position containing -1 with nan value
    data_initial[data_initial == -1] = np.nan

    # Random shuffling of dataset
    np.random.shuffle(data_initial)

    # Rating <=4 is negative and >=7 is positive
    label = np.array(data_initial[:,0] >= 7)*1
    data_initial = data_initial[:,1:]
    
    Y = label.reshape(label.shape[0], 1)
    X = data_initial

    return X, Y

def data_load_spamassasin(data_folder):
    data_name = "spamassasin.pickle"
    data_path = data_folder_path(data_folder, data_name)
    # To load the file
    with open(data_path, 'rb') as handle:
        data_initial = pickle.load(handle)

    data_initial = data_initial.astype(float)
    # Substitute each position containing -1 with nan value
    data_initial[data_initial == -1] = np.nan
    
    Y = data_initial[:,0:1]
    X = data_initial[:, 1:]

    return X, Y

def data_load_crowdsense_c3(data_folder):
    data_name = "crowdsense_c3.pickle"
    data_path = data_folder_path(data_folder, data_name)
    # To load the file
    with open(data_path, 'rb') as handle:
        data_initial = pickle.load(handle)

    data_initial = data_initial.astype(float)
    # Substitute each position containing -1 with nan value
    data_initial[data_initial == -1] = np.nan
    
    Y = data_initial[:,0:1]
    X = data_initial[:, 1:]

    return X, Y

def data_load_crowdsense_c5(data_folder):
    data_name = "crowdsense_c5.pickle"
    data_path = data_folder_path(data_folder, data_name)
    # To load the file
    with open(data_path, 'rb') as handle:
        data_initial = pickle.load(handle)

    data_initial = data_initial.astype(float)
    # Substitute each position containing -1 with nan value
    data_initial[data_initial == -1] = np.nan
    
    Y = data_initial[:,0:1]
    X = data_initial[:, 1:]

    return X, Y