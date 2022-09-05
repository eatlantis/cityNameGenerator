from .model_path import CITY_NAME_GENERATOR
import numpy as np
import keras


class CityNameGenerator:
    def __init__(self):
        self.model = keras.models.load_model(CITY_NAME_GENERATOR)

    def predict(self, city_text):
        prediction = self.model.predict(np.array([city_text]), verbose=0)[0]

        prediction_int = np.argmax(prediction)
        return prediction_int
