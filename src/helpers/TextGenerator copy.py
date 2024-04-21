import tensorflow as tf
import numpy as np

from src.data.Text.Tokens.BaseClass import TokenBaseClass



"""
    1- Generated Text: Tracks text generated by the model and will maintain the output to be passed back into the next iteration of the model
       Deals with tokeniser of the output of the model 
       (takes into account the length of the content to ensure that there is no overflow being passed into model evaluation call)

    2- Input Handler: Takes input either as string or token and creates input to the model 
        (different sublasses account for different expectations in the formatting of the data assumed by the model)
        (Deals with tokeniser for all domains)
    
    3- Text Generator: Takes model and input handler to generate text, responsible for recursive calling of model and using passed classes to keep consistent ouptut
       (a) Text input creates initial input to model, also initializes initial value of Generated text object
       (b) Model pass performed
       (c) New token passed to Generated Text option
       (d) Updated output tensor passed back to input for next iteration


"""


def string_to_tokens(input_string: str, tokeniser: TokenBaseClass, seq_len=None):
    tokenised_text = tokeniser.tokenise(input=[input_string], sequence_length=seq_len) #This returns slice dataset (can't be fed directly into model)
    numpy_text = [i.numpy() for i in tokenised_text]
    tensor_obj = tf.convert_to_tensor(numpy_text[0])
    return tensor_obj

class GeneratedText:
    def __init__(self, initial_tokens: tf.Tensor, output_tokeniser: TokenBaseClass, sequence_fixed_size: int= None):
        """
            Sequence max len will determine if the runing list needs to be cut up (this will depend on the architecture)
            For example, if the model expects a certain size, the running list should not be permitted to exceed that size
        """
        self.initial_tokens = initial_tokens
        self.output_tokeniser = output_tokeniser
        self.generated_tokens = tf.convert_to_tensor([], dtype=tf.int32)
        self.token_list_for_model = self.initial_tokens#This is the variable that will be fed into the model

        self.sequence_fixed_size = sequence_fixed_size

    def add_new_token(self, new_token:int):
        self.generated_tokens = tf.concat([self.generated_tokens, [new_token]], axis=0)
        self.format_tensor_for_model()

    def format_tensor_for_model(self):
        self.token_list_for_model = tf.concat([self.initial_tokens, self.generated_tokens], axis=0)
        if self.sequence_fixed_size is None: return #no further trimming required, can pass through the entire tensor
        self.token_list_for_model = self.token_list_for_model[-1*self.sequence_fixed_size:]



class ModelInputHandle:
    def __init__(self):
        pass
        #Output initializer
        #Prepare initial inputs(in case it's given a string(s))


from typing import Union

class TextGenerator:
    def __init__(self, domain_count: int, inputs: Union[tf.Tensor, list], input_model):
        self.input_handler = input_handler
        self.input_model = input_model

        self.initial_output = self.input_handler.get_initial_output()
        self.output_tokeniser = self.input_handler.get_output_tokeniser()
        self.generated_text = GeneratedText(self.initial_output,)









"""
from typing import Protocol, Union


class InputHandler(Protocol):
    def convert_initial_input(self):
        ...


    def load_in_generated_tokens(self, input_tokens: tf.Tensor):
        ...


class StandardRecursionInput:
    def __init__(self, input: Union[str, tf.Tensor], input_tokeniser: TokenBaseClass, seq_len = None):
        self.input = input
        self.input_tokeniser = input_tokeniser
        self.seq_len = seq_len
        self.model_input = self.convert_initial_input()

    def convert_initial_input(self) -> tf.Tensor:
        output = string_to_tokens(self.input, self.input_tokeniser, self.seq_len) if isinstance(self.input, str) else self.input
        return output

    def load_in_generated_tokens(self, input_tokens: tf.Tensor) -> None:
        #this is the text from the generated text object
        self.model_input = input_tokens


class TransformerRecursionInput:
    def __init__(self, context_input: Union[str, tf.Tensor], context_tokeniser: TokenBaseClass
                 , content_input: Union[str, tf.Tensor], content_tokeniser: TokenBaseClass
                 , context_seq_len: int):
        #Context Variables (String input)
        self.context_input = context_input
        self.context_tokeniser = context_tokeniser
        self.context_input_object = StandardRecursionInput(self.context_input, self.context_tokeniser, context_seq_len)
        self.context_tokens = self.context_input_object.convert_initial_input()

        #Content Variables (Where output is handeled)
        self.content_input = content_input
        self.content_tokeniser = content_tokeniser
        self.content_input_object = StandardRecursionInput(self.content_input, self.content_tokeniser)

        #Inputs
        self.model_input = self.convert_initial_input()


    def convert_initial_input(self) -> tf.Tensor:
        output = tf.concat([self.context_tokens, self.context_input_object.convert_initial_input()], axis=0)
        return output
        
    def  load_in_generated_tokens(self, input_tokens: tf.Tensor) -> None:
        self.model_input = tf.concat([self.context_tokens, input_tokens], axis=0)


"""

"""
    1- Generated Text: Tracks text generated by the model and will maintain the output to be passed back into the next iteration of the model
       Deals with tokeniser of the output of the model 
       (takes into account the length of the content to ensure that there is no overflow being passed into model evaluation call)

    2- Input Handler: Takes input either as string or token and creates input to the model 
        (different sublasses account for different expectations in the formatting of the data assumed by the model)
        (Deals with tokeniser for all domains)
    
    3- Text Generator: Takes model and input handler to generate text, responsible for recursive calling of model and using passed classes to keep consistent ouptut
       (a) Text input creates initial input to model, also initializes initial value of Generated text object
       (b) Model pass performed
       (c) New token passed to Generated Text option
       (d) Updated output tensor passed back to input for next iteration


"""

class TextGenerator:
    def __init__(self, input_handler: InputHandler, input_model):
        self.input_handler = input_handler
        self.input_model = input_model
        self.text_generator = GeneratedText()







from src.data.Text.Tokens.CustomTokenisers import CustomCharacterToken
input_text = "abcdefghijklmnopqrstuvqwxyz "

B = CustomCharacterToken(100)
B.generate_vocab(input_text)

text_gen = GeneratedText(tf.convert_to_tensor([1]), B)





#for i in range(20): text_gen.add_new_token(i)

"""



A = TokenStringContainer()
A.load_inputs_by_string("hello there how is it", B, 50)

C = GenTokenTracker(A.token_txt, 0)
C.add_new_token(15)
"""

#D = GeneratedTextTracker([0], 100)













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



"""

class TokenStringContainer:
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



"""



























"""






class SingleInputHandler:
    """
        Responsible for sinlgle domain of input
        Create string and token representations
        Can update input tokens for recursive
    """
    def __init__(self, input: Union[str, tf.Tensor], input_tokeniser: TokenBaseClass, seq_len = None):
        self.token_input = None
        self.update_input(input, input_tokeniser, seq_len)
        self.tokeniser = input_tokeniser

    def update_input(self, input: Union[str, tf.Tensor], input_tokeniser: TokenBaseClass, seq_len = None):
        output = string_to_tokens(input, input_tokeniser, seq_len) if isinstance(input, str) else input
        self.token_input = output






class GeneralInputHandler:
    def __init__(self, string_beginning: Union[str, tf.Tensor], output_tokeniser: TokenBaseClass
                 , input_model
                 , context_inputs: Union[str, list] = None, context_tokeniser: Union[str, list] = None):
        
        self.auto_regressive_input = SingleInputHandler(input=string_beginning, input_tokeniser=output_tokeniser)
        self.input_model = input_model
        self.context_inputs = self.handle_context_inputs(context_inputs, context_tokeniser)
        self.single_domain = True if self.context_inputs is None else False
        self.token_input = self.combine_domains()
    
    def update_input(self, input: Union[str, tf.Tensor], input_tokeniser: TokenBaseClass, seq_len = None):
            self.auto_regressive_input.update_input(input=input, input_tokeniser=input_tokeniser, seq_len=seq_len)
            self.token_input = self.combine_domains()
    
    def combine_domains(self):
        if self.single_domain: return self.auto_regressive_input.token_input
        all_inputs = self.context_inputs.append(self.auto_regressive_input)
        return tf.stack([x.token_input for x in all_inputs], axis=1)

    
    def handle_context_inputs(self, context_inputs: Union[str, list] = None, context_tokeniser: Union[str, list] = None) ->list:
        if context_inputs is None: return #there is no context

        #Ensure all in list format
        if isinstance(context_inputs,str):
            context_inputs = [context_inputs]
            context_tokeniser = [context_tokeniser]

        zipped_inputs = zip(context_inputs, context_tokeniser)
        output = [SingleInputHandler(input=input, input_tokeniser=tokeniser, seq_len=tokeniser.sequence_len) 
                  for input, tokeniser in zipped_inputs]

        return output




class ModelHandle:
    def __init__(self, input_model):
        self.input_model = input_model

    def get_output(self, model_input):
        raw_output = self.input_model(model_input)
        return raw_output


    def get_max_token(self, model_output, index):
        pass






"""