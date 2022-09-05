import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from A_dataset.get_dataset import get_dataset
from D_model.model import get_model
import numpy as np

max_encoding_key, labels, data_columns = get_dataset()

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

model_name = 'test1'
model_folder = 'test' if 'test' in model_name else 'main'

model_address = f'../model_store/{model_folder}/{model_name}.hdf5'
log_address = f'./logs/{model_name}.csv'

model_checkpoint = ModelCheckpoint(model_address,
                                   monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor=f'val_loss', patience=15, min_delta=0.0001)
csv_logger = CSVLogger(log_address, append=True, separator=',')

model = get_model(max_encoding_key)
model.fit(x_train, y_train, epochs=150, batch_size=64, steps_per_epoch=10,
          callbacks=[model_checkpoint, early_stopping, csv_logger])

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
