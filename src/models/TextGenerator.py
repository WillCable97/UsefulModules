import tensorflow as tf
import numpy as np


class TextGenerator:
    def __init__(self, input_model, output_token):
        self.input_model = input_model
        self.output_token = output_token

    def load_inputs_by_string(self, input:list, tokenisers):
        self.string_inputs = input
        token_inputs = [tokenisers[i].tokenise([string_val]) for i, string_val in enumerate(input)]
        self.token_inputs = token_inputs
    
    def load_inputs_by_token(self, input: list, tokenisers): #This is going to expect Tensor objects as input
        self.token_inputs = input#tf.squeeze(input, axis = 0)
        string_inputs = [tokenisers[i].detokenise(tok_seq) for i, tok_seq in enumerate(input)]
        self.string_inputs = string_inputs

    def generate_text(self, max_len:int, content_input_index = 0, termination_token = None):
        generated_tokens = []
        starting_text = self.string_inputs[content_input_index]
        sliding_token_list = self.token_inputs

        for i in range(max_len):
              model_output = self.input_model(sliding_token_list)
              gen_distribution = model_output[0][-1]
              found_token = tf.argmax(gen_distribution)
              sliding_token_list[content_input_index] = np.hstack((sliding_token_list[content_input_index], [[found_token]]))
              sliding_token_list[content_input_index] = sliding_token_list[content_input_index][:, 1:]
              generated_tokens.append(found_token)

        generated_tokens = tf.convert_to_tensor([generated_tokens])
        print(starting_text)
        return ''.join(self.output_token .detokenise(generated_tokens))
