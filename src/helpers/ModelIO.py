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
    tf.data.experimental.save(validation_data, os.path.join(model_data, "validation_data"))

    model_object.save(os.path.join(model_objs, "model"))
    create_dir(os.path.join(model_objs, "tokens"))

    for token in tokenisers: 
        token_path = os.path.join(model_objs, "tokens", f"{token}.pkl")
        token_file = open(token_path, "wb")
        pickle.dump(tokenisers[token], token_file)

    

class ModelFromFile:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def perform_load(self):
        base_path = model_path(self.model_name)
        model_data = os.path.join(base_path, "data")
        model_objs = os.path.join(base_path, "model_objs")

        training_data = tf.data.experimental.load(os.path.join(model_data, "training_data"))
        validation_data = tf.data.experimental.load(os.path.join(model_data, "validation_data"))
        model_obj = tf.keras.models.load_model(os.path.join(model_objs, "model"))
        base_token_path = os.path.join(model_objs, "tokens")
        tokens = {}

        for token_file in os.listdir(base_token_path):
            split_name = token_file.split(".")
            token_name = split_name[0]
            full_token_file = open(os.path.join(base_token_path, token_file), "rb")
            token_obj = pickle.load(full_token_file)
            tokens[token_name] = token_obj

        self.training_data = training_data
        self.validation_data = validation_data
        self.model_obj = model_obj
        self.tokenisers = tokens


    

