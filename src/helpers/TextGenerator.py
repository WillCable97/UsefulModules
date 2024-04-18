import tensorflow as tf
import numpy as np

from src.data.Text.Tokens.BaseClass import TokenBaseClass


class TokenStringContainer:
    """
        An object that takes either a string and converts to a tensor
        Or and takes a tensor and converts to a string
        Provides easy conversion between forms when analysing outputs
    """
    def __init__(self):
        self.padded_sequence_lens: int = None
        self.string_txt: str = None
        self.token_txt: tf.Tensor = None

    def load_inputs_by_string(self, input_string: str, tokeniser: TokenBaseClass, seq_len=None):
        self.string_txt = input_string
        tokenised_text = tokeniser.tokenise(input=[input_string], sequence_length=seq_len) #This returns slice dataset (can't be fed directly into model)
        numpy_text = [i.numpy() for i in tokenised_text]
        tensor_obj = tf.convert_to_tensor(numpy_text[0])
        self.token_txt = tensor_obj

class GenTokenTracker:
    """
        Keeps track of the generated tokenss will account for padding 
        so that generated tokens are added the end of the actual sequence
    """

    def __init__(self, initial_tokens: tf.Tensor, padding_token: int):
        self.initial_tokens = initial_tokens
        self.running_seq = self.unpad_sequence(self.initial_tokens, padding_token)

    def unpad_sequence(self, input_seq:tf.Tensor, padding_token: int) -> tf.Tensor:
        is_padding = (input_seq==padding_token)
        indexes_of_padding = tf.where(is_padding)
        initial_padding_index = None if tf.size(indexes_of_padding)==0 else indexes_of_padding[0][0].numpy()
        return input_seq[:initial_padding_index]
    
    def add_new_token(self, new_token: int):
        self.running_seq = tf.concat([self.running_seq, [new_token]], axis=0)




class GeneratedTextTracker:
    def __init__(self, initial_tokens: tf.Tensor, text_tokeniser: TokenBaseClass):
        self.initial_token = initial_tokens
        self.text_tokeniser = text_tokeniser

        self.generated_tokens = tf.convert_to_tensor([])
        self.running_sequence = self.initial_token

        self.max_seq_len = text_tokeniser.sequence_len

    def add_new_token(self, new_token:int):
        






from src.data.Text.Tokens.CustomTokenisers import CustomCharacterToken
input_text = "abcdefghijklmnopqrstuvqwxyz "

B = CustomCharacterToken(100)
B.generate_vocab(input_text)

A = TokenStringContainer()
A.load_inputs_by_string("hello there how is it", B, 50)

C = GenTokenTracker(A.token_txt, 0)
C.add_new_token(15)


D = GeneratedTextTracker([0], 100)













"""

from typing import Protocol

class TextGeneratorProtocol(Protocol):
    def generate_text(self, max_len:int, termination_token=None):
        ...

class TextGeneratorGeneral(TextGeneratorProtocol):
    def __init__(self, input_model, output_token, string_container: TokenStringContainer):
        self.input_model = input_model
        self.output_token = output_token
        self.string_container = string_container
        self.output_tracker = GenTokenTracker(string_container.token_txt, padding_token=0)

    def generate_text(self, max_len:int )



class TextGenerator:
    def __init__(self, input_model, output_token, string_container: TokenStringContainer):
        self.input_model = input_model
        self.output_token = output_token
        self.string_container = string_container

    def generate_text(self, max_len:int)

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
        return ''.join(self.output_token.detokenise(generated_tokens))



"""


























"""
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

"""


        
"""
class TextGenerator:
    def __init__(self, input_model, output_token, padded_sequence_lens):
        self.input_model = input_model
        self.output_token = output_token
        self.padded_sequence_lens = padded_sequence_lens

    def load_inputs_by_string(self, input:list, tokenisers):
        self.string_inputs = input
        formated_input = [[i] for i in input]
        tokenised_text = [tokenisers[i].tokenise(string_val, sequence_length=self.padded_sequence_lens[i]) for i, string_val in enumerate(formated_input)]
        numpy_arrays = [[item.numpy() for item in sliced_dataset] for sliced_dataset in tokenised_text]
        tensor_objects = [tf.convert_to_tensor(i) for i in numpy_arrays]
        print(tensor_objects)
        self.token_inputs = tensor_objects
    
    def load_inputs_by_token(self, input: list, tokenisers): #This is going to expect Tensor objects as input
        self.token_inputs = input
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
        return ''.join(self.output_token.detokenise(generated_tokens))

"""