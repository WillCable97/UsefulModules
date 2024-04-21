from src.helpers.Callbacks.BaseCallbacks import path_to_model_saves
from src.helpers.TextGenerator import TextGenerator
from src.data.Text.Tokens.BaseClass import TokenBaseClass

import os 
#import keras
import tensorflow as tf
from typing import Union




class OutputTextCallback(tf.keras.callbacks.Callback):
    def __init__(self, string_beginning: Union[str, tf.Tensor], output_tokeniser: TokenBaseClass
                 , input_model: tf.keras.Model, base_path: str, input_model_name:str
                 , file_name = 'text_outputs.txt'
                 , context_inputs: Union[str, list] = None, context_tokenisers: Union[TokenBaseClass, list] = None):
        self.generator = TextGenerator(string_beginning=string_beginning, output_tokeniser=output_tokeniser, input_model=input_model
                                       , context_inputs=context_inputs, context_tokenisers=context_tokenisers)
        
        self.base_path = base_path
        self.input_model_name = input_model_name
        self.file_name = file_name   
        self.txt_generated = "" 

        
    def on_epoch_end(self, epoch, logs=None):
        self.generator.model_handle.input_model.set_weights(self.model.get_weights())
        txt = self.generator.generate_sequence(50)
        print(f"\ngenerated text:  {txt}\n")
        self.txt_generated += f"{epoch + 1}: {txt}\n"
        self.write_output_to_file()

    def write_output_to_file(self):
        full_path = path_to_model_saves(self.base_path, self.input_model_name)
        full_path = os.path.join(full_path, "text_tracker")
        if not os.path.exists(full_path): os.makedirs(full_path)
        file_path = os.path.join(full_path, self.file_name)
        file = open(file_path, 'w+')
        file.write(self.txt_generated)