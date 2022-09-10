from masterPathAddress import MASTER_PATH
from tqdm import tqdm
import pandas as pd
import os

data_sources_folder_addr = os.path.join(MASTER_PATH, 'B_data_gathering')
data_sources_folder_contents = os.listdir(data_sources_folder_addr)
file_columns = ['civ', 'city']

civilization_city_names = []


replacements = {
    'ø': 'o',
    'ê': 'e',
    'ü': 'u',
    'æ': 'ae',
    '': 'a',
}
chars_to_remove = [
    '(',
    ')',
    '-',
    "'",
]


def clean_word(list_word):
    list_word = str(list_word).lower()
    if '[' in list_word:
        list_word = list_word.split('[')[0]
    if '(' in list_word:
        list_word = list_word.split('(')[0]

    for repl_w in replacements:
        if repl_w in list_word:
            list_word = list_word.replace(repl_w, replacements[repl_w])
    for rem_w in chars_to_remove:
        if rem_w in list_word:
            list_word = list_word.replace(rem_w, '')
    return list_word


for source_name in tqdm(data_sources_folder_contents, desc="Combine Sources"):
    source_addr = os.path.join(data_sources_folder_addr, source_name)

    source_file_name = f'{source_name}.csv'
    source_contents = os.listdir(source_addr)
    if source_file_name in source_contents:
        source_file_addr = os.path.join(source_addr, source_file_name)
        dataset = pd.read_csv(source_file_addr, encoding='latin')
        dataset = dataset.filter(items=file_columns)
        dataset_lists = dataset.values.tolist()
        for d_list in dataset_lists:
            for w_idx, word in enumerate(d_list):
                cleaned_word = clean_word(word)
                d_list[w_idx] = cleaned_word
            civilization_city_names.append(d_list)

dataset_output_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '0_cities_list.csv')
cities_df = pd.DataFrame(civilization_city_names)
cities_df.columns = file_columns
cities_df.to_csv(dataset_output_file_addr, index=False)
