import os
import tensorflow as tf
from keras import Model
import pickle

def model_path(model_name: str) -> str:
    base_bath = os.path.abspath("./")
    model_path = os.path.join(base_bath, "models", model_name)
    return model_path


def create_dir(dir_path: str) -> bool: #Returns true of directory created 
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path)
        return True
    return False

def create_model_save(model_name: str, model_object:Model
                      , training_data, validation_data
                      , tokenisers = {}):
    base_path = model_path(model_name)
    model_data = os.path.join(base_path, "data")
    model_objs = os.path.join(base_path, "model_objs")

    tf.data.experimental.save(training_data, os.path.join(model_data, "training_data"))
    tf.data.experimental.save(training_data, os.path.join(model_data, "validation_data"))

    model_object.save(os.path.join(model_objs, "model"))
    create_dir(os.path.join(model_objs, "tokens"))

    for token in tokenisers: 
        token_path = os.path.join(model_objs, "tokens", f"{token}.pkl")
        token_file = open(token_path, "wb")
        pickle.dump(tokenisers[token], token_file)

    
    