import random

from masterPathAddress import MASTER_PATH
import pandas as pd
import os


dataset_values_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '2_encoded_dataset_values.csv')
dataset_labels_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '2_encoded_dataset_labels.csv')

dataset_values_file = pd.read_csv(dataset_values_file_addr)
dataset_labels_file = pd.read_csv(dataset_labels_file_addr)
dataset_labels_columns = list(dataset_labels_file.columns)

min_optionality = 100000
max_optionality = 0
balanced_labels = []
balanced_values = []

labels_list = dataset_labels_file[dataset_labels_columns[1]].tolist()
different_labels = set(labels_list)

label_rows = {}
rows_values = {}
for label_idx, label in enumerate(labels_list):
    if label not in label_rows:
        label_rows[label] = []
        rows_values[label] = []

    value = dataset_values_file.loc[label_idx].tolist()
    label_rows[label].append(label_idx)
    rows_values[label].append(value)
    label_rows_len = len(label_rows[label])
    if label_rows_len < min_optionality:
        min_optionality = label_rows_len
    if label_rows_len > max_optionality:
        max_optionality = label_rows_len

new_dataset_rows = []
new_label_rows = []
for label in label_rows:
    original_label_row_indexes = label_rows[label]
    label_row_value = label_rows[label]
    num_values = len(label_row_value)
    for _ in range(max_optionality):
        random_index_chosen = random.randint(0, num_values - 1)
        chosen_value = rows_values[label][random_index_chosen]

        new_dataset_rows.append(chosen_value)
        new_label_rows.append(label)

rows_df = pd.DataFrame(new_dataset_rows)
rows_df.to_csv(os.path.join(MASTER_PATH, 'A_dataset', '3_balanced_dataset_values.csv'), index=False)

keys_df = pd.DataFrame(new_label_rows)
keys_df.to_csv(os.path.join(MASTER_PATH, 'A_dataset', '3_balanced_dataset_labels.csv'), index=False)
