from masterPathAddress import MASTER_PATH
import pandas as pd
import os

tokenized_dataset_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '1_tokenized_dataset_values.csv')
tokenized_dataset = pd.read_csv(tokenized_dataset_file_addr)
tokenized_labels_dataset_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '1_tokenized_dataset_labels.csv')
tokenized_labels_dataset = pd.read_csv(tokenized_labels_dataset_file_addr)

if len(tokenized_dataset) != len(tokenized_labels_dataset):
    raise Exception('Stop! Unequal LabelxValue Lengths')

tokens_list = tokenized_dataset.values.tolist()
tokens_labels_list = tokenized_labels_dataset['1'].values.tolist()

token_counts = {}
for t_list in tokens_list:
    for token in t_list:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1

for token in tokens_labels_list:
    if token not in token_counts:
        token_counts[token] = 0
    token_counts[token] += 1

token_counts_lists = [[token, token_counts[token]] for token in token_counts]

token_counts_df = pd.DataFrame(token_counts_lists)
token_counts_df.columns = ['token_value', 'count']

token_counts_df = token_counts_df.sort_values('count', ascending=False)
token_counts_df['token_key'] = [token_index + 0 for token_index in range(len(token_counts_df))]

word_count_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '2a_word_count_file.csv')
token_counts_df.to_csv(word_count_file_addr, index=False)

# Now will replace all tokens in file
token_keys = {token_counts_df.loc[token_index, 'token_value']: token_counts_df.loc[token_index, 'token_key'] for
              token_index in range(len(token_counts_df))}

tokenized_dataset = tokenized_dataset.replace(token_keys, inplace=False)
encoded_dataset_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '2_encoded_dataset_values.csv')
tokenized_dataset.to_csv(encoded_dataset_file_addr, index=False)

tokenized_labels_dataset = tokenized_labels_dataset.replace(token_keys, inplace=False)
encoded_dataset_labels_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '2_encoded_dataset_labels.csv')
tokenized_labels_dataset.to_csv(encoded_dataset_labels_file_addr, index=False)
