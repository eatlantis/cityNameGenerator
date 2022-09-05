import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from A_dataset.get_dataset import get_dataset
from D_model.model import get_model
import numpy as np
import random
import sys

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

model_name = 'test2'
model_folder = 'test' if 'test' in model_name else 'main'

model_address = f'../model_store/{model_folder}/{model_name}.hdf5'
log_address = f'./logs/{model_name}.csv'

model_checkpoint = ModelCheckpoint(model_address,
                                   monitor='loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor=f'loss', patience=15, min_delta=0.0001)
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
    if epoch % 1 == 0:
        print("******************************************************")
        print('----- Generating text after Epoch: %d' % epoch)

        random_city_index = random.randint(0, len(x_test) - 1)
        random_city_line = x_test[random_city_index]

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
                preds = model.predict(np.array([word_list]), verbose=0)[0]
                preds = np.where(preds == np.max(preds), 1, 0)
                word_list.append(preds)

                next_char = encoding_dict[np.argmax(preds)]
                generated += next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x_train, y_train, epochs=150, batch_size=128,
          callbacks=[model_checkpoint, early_stopping, csv_logger, print_callback])

model.save(model_address)

loss_label = 'Mean Squared Error'
metrics = 'accuracy'

loss_chart_name = f'./pics/{model_name}_loss.png'
plt.plot(model.history.history['loss'])
plt.plot(model.history.history[f'val_loss'])
plt.ylabel(loss_label)
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig(loss_chart_name)
plt.show()

acc_chart_name = f'./pics/{model_name}_acc.png'
plt.plot(model.history.history[metrics])
plt.plot(model.history.history[f'val_{metrics}'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig(acc_chart_name)
plt.show()
