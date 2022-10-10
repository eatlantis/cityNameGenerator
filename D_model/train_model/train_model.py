import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from A_dataset.get_dataset import get_dataset
from D_model.model import get_model
from global_vars import INPUT_LEN
import numpy as np
import random
import sys

MAX_LEN = INPUT_LEN


max_encoding_key, labels, data_columns, encoding_dict, rev_encoding_dict = get_dataset()

x_train, x_test, y_train, y_test = train_test_split(data_columns, labels,
                                                    test_size=0.2)
mean_train = y_train.mean()

print('x_train')
print(x_train.shape)
print('y_train.shape')
print(y_train.shape)


# trainable_count = int(
#     np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
# non_trainable_count = int(
#     np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
#
# print('Total params: {:,}'.format(trainable_count + non_trainable_count))
# print('Trainable params: {:,}'.format(trainable_count))
# print('Non-trainable params: {:,}'.format(non_trainable_count))

model_name = 'test12'
model_folder = 'test' if 'test' in model_name else 'main'

model_address = f'../model_store/{model_folder}/{model_name}.hdf5'
log_address = f'./logs/{model_name}.csv'

model_checkpoint = ModelCheckpoint(model_address,
                                   monitor='loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor=f'loss', patience=500
                               , min_delta=0.0001)
csv_logger = CSVLogger(log_address, append=True, separator=',')

model = get_model(max_encoding_key)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    try:
        if epoch % 3 == 0:
            print("******************************************************")
            print('----- Generating text after Epoch: %d' % epoch)

            random_city_index = random.randint(0, len(x_test) - 1)
            random_city_line = x_test[random_city_index]
            blank_text = np.zeros(len(random_city_line[0]))
            blank_text[0] = 1

            start_index = random.randint(0, len(x_test) - max_encoding_key - 1)
            for temperature in [0.2, 0.5, 1.0, 1.2]:
                print('----- temperature:', temperature)

                generated = ''
                sentence = ''.join(encoding_dict[np.argmax(l)] for l in random_city_line)
                generated += sentence
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                word_list = list(random_city_line)

                for i in range(15):
                    entry_word_list = [item for item in word_list]
                    while len(entry_word_list) < MAX_LEN:
                        entry_word_list.insert(0, 0)
                    if len(entry_word_list) > MAX_LEN:
                        entry_word_list = entry_word_list[-MAX_LEN:]
                    preds = model.predict(np.array([entry_word_list]), verbose=0)[0]
                    next_index = sample(preds, temperature)

                    empty_text = np.zeros(len(random_city_line[0]))
                    empty_text[next_index] = 1

                    word_list.append(empty_text)

                    next_char = encoding_dict[next_index]
                    generated += next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()
    except:
        pass


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x_train, y_train, epochs=20000, batch_size=16, steps_per_epoch=100,
          callbacks=[model_checkpoint, early_stopping, csv_logger, print_callback])

model.save(model_address)

loss_label = 'Categorical Crossentropy'
metrics = 'accuracy'

loss_chart_name = f'./pics/{model_name}_loss.png'
plt.plot(model.history.history['loss'])
plt.ylabel(loss_label)
plt.xlabel('Epoch')
plt.legend(['Train'])
plt.savefig(loss_chart_name)
plt.show()

acc_chart_name = f'./pics/{model_name}_acc.png'
plt.plot(model.history.history[metrics])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'])
plt.savefig(acc_chart_name)
plt.show()
