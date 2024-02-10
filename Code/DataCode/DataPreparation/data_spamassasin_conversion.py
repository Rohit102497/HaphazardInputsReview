import glob
import re
from datetime import datetime
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

# -------------------------------
path = "/Users/rag004/Documents/PhD/Code/HaphazardInputReview/HaphazardInputs/Data/"
# Directories
directories = {"easy_ham"    : path + "spamassasin/easy_ham",
               "easy_ham_2"  : path + "spamassasin/easy_ham_2",
               "hard_ham"    : path + "spamassasin/hard_ham",
               "spam"        : path + "spamassasin/spam",
               "spam_2"      : path + "spamassasin/spam_2",
               }

# Date Formats present in emails :

# Wed, 21 Aug 2002 10:54:46 -0200
# Wed, 21 Aug 2002 10:54:46
# Wed, 21 Aug 2002
# 06 Sep 2002 10:13:17 -0700
# 06 Sep 2002 10:13:17
# 06 Sep 2002

# Regular expressions for all date formats
date_regex_list = [
                    # r"[A-Za-z]+, [0-9]+ [A-Za-z]+ [0-9]+ [0-9]+:[0-9]+:[0-9]+ -[0-9][0-9][0-9][0-9]",    # Wed, 21 Aug 2002 10:54:46 -0200
                    r"[A-Za-z]+, [0-9]+ [A-Za-z]+ [0-9]+ [0-9]+:[0-9]+:[0-9]+",                          # Wed, 21 Aug 2002 10:54:46
                    r"[A-Za-z]+, [0-9]+ [A-Za-z]+ [0-9]+",                                               # Wed, 21 Aug 2002
                    # r"[0-9]+ [A-Za-z]+ [0-9]+ [0-9]+:[0-9]+:[0-9]+ -[0-9][0-9][0-9][0-9]",               # 06 Sep 2002 10:13:17 -0700
                    r"[0-9]+ [A-Za-z]+ [0-9]+ [0-9]+:[0-9]+:[0-9]+",                                     # 06 Sep 2002 10:13:17
                    r"[0-9]+ [A-Za-z]+ [0-9]+",                                                          # 06 Sep 2002
                   ]

# Date formats to extract datetime object
date_format_list = [
                    # "%a, %d %b %Y %H:%M:%S %z", # Wed, 21 Aug 2002 10:54:46 -0200
                    # "%a, %d %b %y %H:%M:%S %z", # Wed, 21 Aug 02 10:54:46 -0200
                    "%a, %d %b %Y %H:%M:%S",    # Wed, 21 Aug 2002 10:54:46
                    "%a, %d %b %y %H:%M:%S",    # Wed, 21 Aug 02 10:54:46
                    "%a, %d %b %Y",             # Wed, 21 Aug 2002
                    "%a, %d %b %y",             # Wed, 21 Aug 02
                    # "%d %b %Y %H:%M:%S %z",     # 21 Aug 2002 10:54:46 -0200
                    # "%d %b %y %H:%M:%S %z",     # 21 Aug 02 10:54:46 -0200
                    "%d %b %Y %H:%M:%S",        # 21 Aug 2002 10:54:46
                    "%d %b %y %H:%M:%S",        # 21 Aug 02 10:54:46
                    "%d %b %Y",                 # 21 Aug 2002
                    "%d %b %y",                 # 21 Aug 02
                    ]

# We decided to not include the timezone-offsets as offset-naive datetime objects and offset-aware datetime objects cannot be compared.

# ------------------------------- Utility Functions
def load_dataset(dirpath: str) -> List[str]:
    """load emails from the specified directory"""
    
    files = []
    filepaths = glob.glob(dirpath + '/*')
    print("Loading Dataset:")
    for path in tqdm(filepaths):
        with open(path, 'rb') as f:
            byte_content = f.read()
            str_content = byte_content.decode('utf-8', errors='ignore')
            files.append(str_content)
    return files

def is_url(s: str) -> bool:
    url = re.match("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", s)
    return url is not None

def str_to_date(date_str: str, date_format_list: List[str]) -> datetime:
    for date_format in date_format_list:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            pass
    return None

def find_last_instance(regex, text):
    matches = list(re.finditer(regex, text))
    if matches:
        last_match = matches[-1]
        return last_match

    return None

# Extract date and body
def get_date_and_body(email: str, date_regex_list: List[str]) -> Tuple[ Union[datetime, None], str ]:
    for date_regex in date_regex_list:
        match_object = find_last_instance(date_regex, email)

        if match_object is not None:
            start, end = match_object.span()
            date_str = date_str = email[start:end]
            date = str_to_date(date_str, date_format_list)
            body = email[end:]
            return date, body
    
    date = None
    body = email 
    return date, body

# Convert URL to word
def convert_url_to_word(words: List[str]) -> List[str]:
    """convert all urls in the list to the word 'URL'"""
    for i, word in enumerate(words):
        if is_url(word):
            words[i] = 'URL'
    return words

# Convert Num to Word
def convert_num_to_word(words: List[str]) -> List[str]:
    """convert all numbers in the list to the word 'NUM'"""
    for i, word in enumerate(words):
        if word.isdigit():
            words[i] = 'NUM'
    return words

# Remove Punctuations
def remove_punctuations(body: str) -> str:
    new_body = ""
    for c in body:
        if c.isalnum() or c.isspace():
            new_body += c
    return new_body

# The cleanup function takes up a dataset (list of email strings) and does the following:

# 1) Extract the date and body of the email and convert the date to a datetime object.
# 2) Converts the body to lowercase
# 3) Replaces every url to the string 'URL'
# 4) Replaces every number to the string 'NUM'
# 5) Removes punctuations

# Finally we return a list of tuples of the form (Date, Body)

def clean_emails(list_of_emails: List[str], 
                 to_lowercase:   bool=True, 
                 url_to_word:    bool=True, 
                 num_to_word:    bool=True, 
                 remove_punc:    bool=True) -> List[Tuple[ Union[datetime, None], str ]]:
    
    cleaned = []
    print("Cleaning Data:")
    for email in tqdm(list_of_emails):
        # 1. Extract date and body
        date, body = get_date_and_body(email, date_regex_list)

        # 2.  Convert to lowercase
        if to_lowercase:
            body.lower()
        
        words = body.split()
        # 3. Convert URL to word
        if url_to_word:
            words = convert_url_to_word(words)
        
        # 4. Convert Num to Word
        if num_to_word:
            words = convert_num_to_word(words)
        
        body = ' '.join(words)

        # 5.  Puntuations
        if remove_punc:
            body = remove_punctuations(body)
        
        cleaned.append((date, body))
    return cleaned

# -------------------------------
# Load the raw datasets and store them in 'datasets' dictionary
# Clean them, convert them into dataframes and store them in the 'dataframes' dictionary

# The dataframes contain 2 columns, 'Date' and 'Body'
# 'Date' column contains the date as a datetime object
# 'Body' contains the body of the email as a string, starting after the date till the end.

datasets = {}
dataframes = {}

for dir_, path in directories.items():

    print(f"=== {dir_} ===")
    data = load_dataset(path)
    data_cleaned = clean_emails(data)
    df = pd.DataFrame(data_cleaned, columns=["Date", "Body"], index=None)

    datasets[f"{dir_}"] = data
    dataframes[f"{dir_}"] = df
    print()

# Print the index and body of the emails, whose date cannot be extracted
for dir_, df in dataframes.items():
    empty_date_rows = df[df['Date'].isna()].reset_index()

    # Display the result
    print(f"=== {dir_} ===")
    for index, body in zip(empty_date_rows["index"], empty_date_rows["Body"]):
        print(f'{index} : {body}')
    print("\n")

# We see these emails does not have any valuable info thus we decide to drop them
for df in dataframes.values():
    df.dropna(subset=['Date'], inplace=True)

# Append class column in the datafames.
for dir_ in ["easy_ham", "easy_ham_2", "hard_ham"]:
    dataframes[dir_]["Y"] = 0
for dir_ in ["spam", "spam_2",]:
    dataframes[dir_]["Y"] = 1

# Concatenate all the dataframes and sort by 'Date'
Df = pd.concat(dataframes.values(), ignore_index=True)
Df = Df.sort_values(by='Date')

# Vectorize the 'string' Body and add it to the df in the 'Vector' column.
vectorizer = CountVectorizer()
vector_col = vectorizer.fit_transform(Df["Body"].tolist()).toarray()
print(f'Number of features per email: {vector_col.shape[1]}')

# Average number of times each feature occurs
avg = np.mean(vector_col, axis=0).squeeze()

# # Each value corresponding to the number of times a feature(word) occurs in the mail
# # We consider the top n=7500 most occuring words as features.

n=7500
top_indices = np.argsort(avg)
top_indices = top_indices[-n:]
print(f'taking top {len(top_indices)} features')

vector_col = vector_col[:, top_indices]
# This takes unnecessarily long (~ 13 mins) when using top 7500 features ??

Df["Vector"] = [vector_col[i, :] for i in range(len(Df))]

# Let's drop the columns we don't want anymore
Df.drop(columns=['Date', 'Body'], inplace=True)

dfs = []
for i in range(Df["Vector"].shape[0]):
    df = pd.DataFrame(Df["Vector"][i]).T
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

merged.insert(loc=0, column='Y', value=Df["Y"])

# -------------------------------
__file__ = os.path.abspath('')
# Save the dataframe in the .pickle file for easy loading
data_frame_arr = np.array(merged)
data_save_path = os.path.join(os.path.dirname(__file__), 'Data',
                          'spamassasin/spamassasin.pickle')
with open(data_save_path, 'wb') as handle:
    pickle.dump(data_frame_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # To load the file
# with open(data_save_path, 'rb') as handle:
#     data_new = pickle.load(handle)
