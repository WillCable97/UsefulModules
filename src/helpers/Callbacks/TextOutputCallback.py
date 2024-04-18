from src.helpers.TextGenerator import TextGenerator
from src.helpers.Callbacks.BaseCallbacks import path_to_model_saves

import os 
import keras



class OutputTextCallback(keras.callbacks.Callback):
    def __init__(self, base_path, input_model_name
                 , input_model, output_token
                 , texts, tokens, padded_sequence_lens
                 , file_name = 'text_outputs.txt'):
        super().__init__()
        self.input_generator = TextGenerator(input_model=input_model, output_token=output_token, padded_sequence_lens=padded_sequence_lens)
        self.input_generator.load_inputs_by_string(texts, tokens)
        self.base_path = base_path
        self.input_model_name = input_model_name
        self.file_name = file_name
        self.txt_generated = ''

    def on_epoch_end(self, epoch, logs=None):
        self.input_generator.input_model.set_weights(self.model.get_weights())
        txt = self.input_generator.generate_text(max_len=50)
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