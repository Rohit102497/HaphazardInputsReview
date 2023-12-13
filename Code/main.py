#--------------Libraries--------------#
import os
import argparse
import random
import numpy as np

#--------------Import Functions--------------#


#--------------All Variables--------------#
if __name__ == '__main__':
    __file__ = os.path.abspath('')
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default="noassumption", type=str, help='The type of the experiment. type options are - "noassumption", "basefeatures", "bufferstorage"')
    parser.add_argument('--data_name', default="noassumption", type=str, help='The type of the experiment. type options are - "noassumption", "basefeatures", "bufferstorage"')
    
    args = parser.parse_args()
    type = args.type

    print("Type: ", type)

#--------------SeedEverything--------------#


#--------------Load Data wrt Variables--------------#


#--------------Load Method wrt Variables--------------#


#--------------Run Model--------------#


#--------------Calculate all Metrics--------------#


#--------------Store results and all variables--------------#