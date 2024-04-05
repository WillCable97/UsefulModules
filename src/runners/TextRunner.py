import helpers
import keras

from src.data.DataObjects.Text.TextDataObjectBase import TextDataObjectBase


class TextModelRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name


    def init_new_model(self, input_data_object: TextDataObjectBase, input_model: keras.Model):
        self.data_object = input_data_object
        self.model = input_model
        self.create_environment(self.model_name)




    def create_environment(self, model_name: str) -> None:
        model_path = helpers.model_path(model_name)
        folder_made = helpers.create_dir(model_path)
        if not folder_made: print("WARNING: New folder not made for model, already exists")


     
A = TextModelRunner("MyModel")