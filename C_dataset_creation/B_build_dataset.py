from masterPathAddress import MASTER_PATH
from model_tools import ModelTools
from global_vars import INPUT_LEN
from tqdm import tqdm
import pandas as pd
import random
import os

NUM_CITIES_GRABBED_RANGE = [1, 5]
CITIES_USED_PER_CIVILIZATION = 500
MAX_LEN = INPUT_LEN

cities_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '0_cities_list.csv')
cities_file = pd.read_csv(cities_file_addr)

civilizations = set(cities_file['civ'].tolist())

civilizations_cities = {}
for civilization in civilizations:
    cities_matches = cities_file[cities_file['civ'] == civilization]['city'].tolist()
    civilizations_cities[civilization] = cities_matches

tokenized_labels = []
tokenized_values = []
for civ in tqdm(civilizations_cities, desc='Adding Civilization'):
    this_civ_cities = civilizations_cities[civ]
    if len(this_civ_cities) > 10:
        for _ in tqdm(range(CITIES_USED_PER_CIVILIZATION), desc='Adding City Data'):
            random_num_of_cities = random.randint(NUM_CITIES_GRABBED_RANGE[0], NUM_CITIES_GRABBED_RANGE[1])
            rand_city_list = []
            while len(rand_city_list) < (random_num_of_cities + 1):
                random_city_index = random.randint(0, len(this_civ_cities) - 1)
                random_city_name = this_civ_cities[random_city_index]
                if str(random_city_name) != 'nan':
                    if str(random_city_name) != 'nan':
                        rand_city_list.append(random_city_name)

            city_text = ','.join(n for n in rand_city_list)

            # Chooses a random token for the label, crops up until that point
            tokenized_value = ModelTools.tokenize_text(city_text, MAX_LEN, ind_chars=True)
            num_tokens = len(tokenized_value)

            label_index = random.randint(int(num_tokens * 0.5), num_tokens - 1)
            label = tokenized_value[label_index]

            if len(tokenized_values) != len(tokenized_labels):
                print('stop')

            tokenized_value = tokenized_value[0: label_index]
            while len(tokenized_value) < MAX_LEN:
                tokenized_value.insert(0, '')

            if len(tokenized_value) > MAX_LEN:
                tokenized_value = tokenized_value[-MAX_LEN:]

            tokenized_values.append(tokenized_value)
            tokenized_labels.append([civ, label])

tokenized_values_df = pd.DataFrame(tokenized_values)
tokenized_values_df_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '1_tokenized_dataset_values.csv')
tokenized_values_df.to_csv(tokenized_values_df_file_addr, index=False)

tokenized_labels_df = pd.DataFrame(tokenized_labels)
tokenized_labels_df_file_addr = os.path.join(MASTER_PATH, 'A_dataset', '1_tokenized_dataset_labels.csv')
tokenized_labels_df.to_csv(tokenized_labels_df_file_addr, index=False)
