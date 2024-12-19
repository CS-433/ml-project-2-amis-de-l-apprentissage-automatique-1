import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, required=True, help="The horizon value to use for extracting the target (1, 2, 3, 5, or 10).")

args = parser.parse_args()
horizon = args.horizon

# index of column depending on wanted horizon
horizon_label = {10: -1, 5: -2, 3: -3, 2: -4, 1: -5}

if horizon not in horizon_label:
    raise ValueError(f"Invalid horizon value: {horizon}. Please choose from {list(horizon_label.keys())}.")

window_size = 10

df_test = pd.read_csv("../test.csv", header=None) # path to train set
df_train = pd.read_csv("../train.csv", header=None) # path to test set

# extracting feature values in train and test set
df_modified_test = df_test.iloc[:, :-5]
df_modified_train = df_train.iloc[:, :-5]
df_combined = pd.concat([df_modified_train, df_modified_test], ignore_index=True)
df_combined.to_csv('./dataset/train_test_modified.csv', index = False) # combine both train and test set in one file and save in data folder

df_train_target = df_train.iloc[:, -5:] # extact all column labels for train and test set
df_test_target = df_test.iloc[:, -5:]

df_targets = pd.concat([df_train_target, df_test_target], ignore_index=True)

df_targets.iloc[:, horizon_label[horizon]].to_csv('./targets/targets.csv', index = False) # saving target file in targets folder