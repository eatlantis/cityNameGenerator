from masterPathAddress import MASTER_PATH
import os

ACTIVE_MODEL = 'test10.hdf5'


model_folder = 'test' if 'test' in ACTIVE_MODEL else 'main'

CITY_NAME_GENERATOR = os.path.join(MASTER_PATH, 'D_model', 'model_store', model_folder, ACTIVE_MODEL)
