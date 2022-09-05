from masterPathAddress import MASTER_PATH
import pandas as pd
import os

data_sources_folder_addr = os.path.join(MASTER_PATH, '1_data_gathering')
data_sources_folder_contents = os.listdir(data_sources_folder_addr)

civilization_city_names = []

for source_name in data_sources_folder_contents:
    source_addr = os.path.join(data_sources_folder_addr, source_name)

    source_file_name = f'{source_name}.csv'
    source_contents = os.listdir(source_addr)
    if source_file_name in source_contents:
        source_file_addr = os.path.join(source_addr, source_file_name)
        dataset = pd.read_csv(source_file_addr)
        dataset_lists = dataset.values.tolist()
        for d_list in dataset_lists:
            civilization_city_names.append(d_list)

dataset_output_file_addr = os.path.join(MASTER_PATH, '0_dataset', '0_cities_list.csv')
cities_df = pd.DataFrame(civilization_city_names)
cities_df.columns = ['civ', 'city']
cities_df.to_csv(dataset_output_file_addr, index=False)
