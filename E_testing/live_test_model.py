from A_dataset.get_dataset import prepare_line
from model_tools import ModelTools
from D_model.get_model import CityNameGenerator

from masterPathAddress import MASTER_PATH
from global_vars import INPUT_LEN
import pandas as pd
import numpy as np
import sys
import os

MAX_LEN = INPUT_LEN
word_encodings_addr = os.path.join(MASTER_PATH, 'A_dataset', '2a_word_count_file.csv')
word_encodings = pd.read_csv(word_encodings_addr)
word_encodings = word_encodings.replace({np.nan: ''}, inplace=False)
encoding_dict = {word_encodings.loc[row_index, 'token_key']: word_encodings.loc[row_index, 'token_value']
                 for row_index in list(word_encodings.index)}
encoding_dict[0] = ''
rev_encoding_dict = {encoding_dict[key]: key for key in encoding_dict}
max_encoding_key = max(word_encodings['token_key'])

name_generator = CityNameGenerator()

print('What temperature do you want')
set_temp = input()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


while True:
    print('\nInsert cities')
    cities_list = input()

    tokenized_cities = ModelTools.tokenize_text(cities_list, MAX_LEN, ind_chars=True)
    cities_token_values = [rev_encoding_dict[letter] for letter in tokenized_cities]
    model_input = list(prepare_line(cities_token_values, is_label=False, two_d=True, max_values=max_encoding_key))

    generated = cities_list
    print(f'Input: {cities_list}\n')
    print(generated)

    for _ in range(20):
        if len(model_input) > MAX_LEN:
            model_input = model_input[-MAX_LEN:]
        prediction = name_generator.predict(np.array(model_input))[0]
        next_index = sample(prediction, 1)

        empty_text = np.zeros(len(model_input[0]))
        empty_text[next_index] = 1

        model_input.append(empty_text)

        next_char = encoding_dict[next_index]
        generated += next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

