import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.data.Preprocessing.Text.Tokenise.BaseClass import TokenBaseClass

class WordTokeniser(TokenBaseClass):
    """
        Wrapper for standard word based tokeniser in Tensorflow. 
        Many of the functions are handled already by the object
        Lightweight wrapper to supper more modularity in training
    """
    def __init__(self, start_string = "[START]", end_string = "[END]"):
        self.core_tokenizer = Tokenizer()
        self.pad_token = 0
        self.start_string = start_string
        self.end_string = end_string

    def generate_vocab(self, input: list) -> None:
        input_with_reserved_tokens = input + [f"{self.start_string} {self.end_string}"]
        
        self.core_tokenizer.fit_on_texts(input_with_reserved_tokens)
        self.vocab = self.core_tokenizer.word_index.keys()
        self.vocab_size = len(self.vocab)

    def tokenise(self, input: list, sequence_length: int = None) -> tf.Tensor:
        mapped_text = self.core_tokenizer.texts_to_sequences(input)       
        padded_text = pad_sequences(mapped_text, padding ="post", maxlen=sequence_length, truncating='post')
        tensor_val = tf.data.Dataset.from_tensor_slices(padded_text)
        return tensor_val 
    
    def detokenise(self, input: tf.Tensor) -> list:
        input = input.numpy()
        print(input)
        ret_list = ["" if token == self.pad_token else f"{self.core_tokenizer.index_word[token]} " for sequence in input for token in sequence]
        return ' '.join(ret_list)