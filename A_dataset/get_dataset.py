from masterPathAddress import MASTER_PATH
import pandas as pd
import numpy as np
import os

encoded_dataset_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '2_encoded_dataset_values.csv')
encoded_dataset_labels_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '2_encoded_dataset_labels.csv')
word_encodings_addr = os.path.join(MASTER_PATH, 'A_dataset', '2a_word_count_file.csv')

MAX_LABEL_LEN = 1
MAX_VALUE_LEN = 70


def get_dataset():
    dataset_labels = pd.read_csv(encoded_dataset_labels_file_addr).values.tolist()
    dataset_values = pd.read_csv(encoded_dataset_file_addr).values.tolist()
    word_encodings = pd.read_csv(word_encodings_addr)

    max_encoding_key = max(word_encodings['token_key'])

    dataset_labels_np = np.array([prepare_line(line, is_label=True, two_d=True, max_values=max_encoding_key)[0]
                                  for line in dataset_labels])
    dataset_values_np = np.array([prepare_line(line, is_label=False, two_d=True, max_values=max_encoding_key)
                                  for line in dataset_values])
    return max_encoding_key + 1, dataset_labels_np, dataset_values_np


def prepare_line(line, is_label=False, two_d=False, max_values=None):
    line_len = MAX_LABEL_LEN if is_label else MAX_VALUE_LEN

    line = list(line)
    while len(line) < line_len:
        line.append(0)

    if two_d is False:
        return np.array(line)

    if max_values is None:
        raise Exception('A_dataset/get_dataset.py '
                        'Must submit variable:max_values to function:prepare_line if you want 2d array')

    empty_array = np.zeros(max_values + 1)
    two_d_line_data = []
    for line_value in line:
        line_value = int(line_value)
        empty_array[line_value] = 1
        two_d_line_data.append(np.array(empty_array))
        empty_array[line_value] = 0
    two_d_line_data = np.array(two_d_line_data)
    return two_d_line_data
