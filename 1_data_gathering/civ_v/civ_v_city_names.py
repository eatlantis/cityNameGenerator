from cities_names_list import CITY_NAME_STR
import pandas as pd
import numpy as np

split_names_list = CITY_NAME_STR.split('\n')
split_names_list = [name for name in split_names_list if name != '']

civ_city_names = []
for name_line in split_names_list:
    name_line = name_line.lower()
    line_split = name_line.split(' ')
    civilization = line_split[0]
    city_name = line_split[1].replace('_', ' ')
    civ_city_names.append([civilization, city_name])

dataframe_df = pd.DataFrame(civ_city_names)
dataframe_df.columns = ['civ', 'city']
dataframe_df.to_csv('ds_civ_v.csv')
