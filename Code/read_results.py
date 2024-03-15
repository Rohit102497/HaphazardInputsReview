import argparse
import pickle
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default="noassumption", type=str,
                        choices = ["noassumption", "basefeatures", "bufferstorage"], 
                        help='The type of the experiment.')
    parser.add_argument('--dataname', default = "wpbc", type = str,
                        choices = ["synthetic", "real", "crowdsense_c5", "crowdsense_c3", "spamassasin", "imdb", "diabetes_us", "higgs", "susy", "a8a", "magic04", 
                                   "spambase", "krvskp", "svmguide3", "ipd", "german", 
                                   "diabetes_f", "wbc", "australian", "wdbc", "ionosphere", "wpbc"],
                        help='The name of the data')
    parser.add_argument('--probavailable', default = 0.5, type = float,
                        help = "The probability of each feature being available to create synthetic data")
    parser.add_argument('--methodname', default = "nb3", type = str,
                        choices = ["nb3", "fae", "olvf", "ocds", "ovfm", 
                                   "dynfo", "orf3v", "auxnet", "auxdrop"],
                        help = "The name of the method")
    
    args = parser.parse_args()
    type = args.type
    data_name = args.dataname
    p_available = args.probavailable
    method_name = args.methodname

    data_name_list = []
    if data_name == "synthetic":
        data_name_list = ["german", "australian", "spambase", "krvskp", "svmguide3", "ipd", 
                        "diabetes_f", "wbc", "wdbc", "ionosphere", "wpbc"] # "magic04", "higgs", "susy", "a8a",   
    elif data_name == "real":
        data_name_list = ["imdb", "diabetes_us", "spamassasin", "crowdsense_c3", "crowdsense_c5"] # "naticusdroid", 
    else:
        data_name_list = [data_name]

    for data_name in data_name_list:

        path_to_result="./Results/"
        result_addr = path_to_result + type + "/" + method_name + "/" + data_name
        df_addr = result_addr

        data_type = "Synthetic"
        if data_name in ["imdb", "diabetes_us", "spamassasin", "naticusdroid", "crowdsense_c3", "crowdsense_c5"]:
            data_type = "Real"

        if data_type == "Synthetic":
            result_addr = result_addr + "_prob_" + str(int(p_available*100)) + ".data"
            df_addr = df_addr + "_prob_" + str(int(p_available*100))
        else:
            result_addr = result_addr + ".data"

        # if method_name == "nb3":
        #     result_addr = result_addr + ".data"

        file = open(result_addr, 'rb') 
        data = pickle.load(file) 

        print("Parameters of this experiment:")
        print(data['params'], "\n")

        # print("All Results: ", pd.DataFrame(data['results']))

        # Calculate mean 
        index = []
        values_list = []
        for i in data['results'].keys():
            index.append(i)
            print(i, pd.DataFrame(data['results'][i]), pd.DataFrame(data['results'][i]).iloc[:, 5:6])
            val_list = pd.DataFrame(data['results'][i]).mean(axis = 0).values.tolist()
            values_list.append(['%.2f' % elem for elem in val_list])
        col_name = list(data['results'][i][0].keys())
        # print(index)
        
        mean_df = pd.DataFrame(values_list, index = index, columns=col_name)
        # print("Mean Values:")
        # print(mean_df)

        # Calculate std deviation 
        index = []
        values_list = []
        for i in data['results'].keys():
            index.append(i)
            val_list = pd.DataFrame(data['results'][i]).std(axis = 0).values.tolist()
            values_list.append(['%.2f' % elem for elem in val_list])
        col_name = list(data['results'][i][0].keys())
        # print(index)
        # print(values_list)

        std_df = pd.DataFrame(values_list, index = index, columns=col_name)
        # print("Std Values:")
        # print(std_df)

        if method_name in ["nb3", "fae", "olvf"]:
            final_df_file = df_addr + 'final.csv'
            print(method_name, ": ", mean_df)
            mean_df.to_csv(final_df_file)
        else:
            final_df = mean_df
            for i in col_name:
                final_df[i] = mean_df[i].astype(str) + '(' + std_df[i].astype(str) + ')'
            final_df_file = df_addr + 'final.csv'
            print(method_name, ": ", final_df)
            final_df.to_csv(final_df_file)
        
        # ["ocds", "dynfo", "orf3v", "auxdrop", "ovfm", "auxnet"]:
        #     final_df = mean_df
        #     for i in col_name:
        #         final_df[i] = mean_df[i].astype(str) + '(' + std_df[i].astype(str) + ')'
        #     final_df_file = df_addr + 'final.csv'
        #     print("ocds final:")
        #     print(final_df)
        #     final_df.to_csv(final_df_file)
        # else:
        #     mean_df_file = df_addr + 'mean.csv'
        #     std_df_file = df_addr + 'std.csv'
        #     mean_df.to_csv(mean_df_file)
        #     std_df.to_csv(std_df_file) 
        