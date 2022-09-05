from masterPathAddress import MASTER_PATH
from model_tools import ModelTools
import pandas as pd
import random
import os

NUM_CITIES_GRABBED_RANGE = [1, 10]
CITIES_USED_PER_CIVILIZATION = 250

cities_file_addr = os.path.join(MASTER_PATH, '0_dataset', '0_cities_list.csv')
cities_file = pd.read_csv(cities_file_addr)

civilizations = set(cities_file['civ'].tolist())

civilizations_cities = {}
for civilization in civilizations:
    cities_matches = cities_file[cities_file['civ'] == civilization]['city'].tolist()
    civilizations_cities[civilization] = cities_matches

tokenized_rows = []
for civ in civilizations_cities:
    this_civ_cities = civilizations_cities[civ]

    for _ in range(CITIES_USED_PER_CIVILIZATION):
        random_num_of_cities = random.randint(NUM_CITIES_GRABBED_RANGE[0], NUM_CITIES_GRABBED_RANGE[1])
        rand_city_list = [civ, ]
        while len(rand_city_list) < (random_num_of_cities + 1):
            random_city_index = random.randint(0, len(this_civ_cities) - 1)
            random_city_name = this_civ_cities[random_city_index]
            if random_city_name not in rand_city_list and str(random_city_name) != 'nan':
                rand_city_list.append(random_city_name)

        city_text = ','.join(n for n in rand_city_list)
        tokenized_text = ModelTools.tokenize_text(city_text)
        tokenized_rows.append(tokenized_text)

tokenized_df = pd.DataFrame(tokenized_rows)
tokenized_df_file_addr = os.path.join(MASTER_PATH, '0_dataset', '1_tokenized_dataset.csv')
tokenized_df.to_csv(tokenized_df_file_addr, index=False)
