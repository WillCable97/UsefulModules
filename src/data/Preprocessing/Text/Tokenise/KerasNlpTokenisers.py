import tensorflow as tf
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras_nlp

from src.data.Preprocessing.Text.Tokenise.BaseClass import TokenBaseClass

class TensorflowWordPieceToken(TokenBaseClass):
    """
        Wrapper for keras Nlp wordpiece tokeniser
        More compute intensive, creating vocaulary is demanding
    """
    def __init__(self, sequence_length: int, max_vocab: int, start_string = "*", end_string = "*"):
        self.pad_token = 0
        self.start_string = start_string
        self.end_string = end_string
        self.sequence_length = sequence_length
        self.vocab_size = max_vocab

    def generate_vocab(self, input: list) -> None:
        tf_inputs = tf.data.Dataset.from_tensor_slices(input)
        reserved_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", self.start_string, self.end_string]
        vocab_list = keras_nlp.tokenizers.compute_word_piece_vocabulary(tf_inputs, self.vocab_size, reserved_tokens=reserved_tokens)
        self.vocab = vocab_list
        self.vocab_size = len(vocab_list)
        self.core_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary = vocab_list, sequence_length = self.sequence_length)

    def tokenise(self, input: list, sequence_length: int = None) -> tf.Tensor:
        return self.core_tokenizer.tokenize(input)
    
    def detokenise(self, input: tf.Tensor) -> list:
        detoken = self.core_tokenizer.detokenize(input)
        interm = detoken.numpy()
        interm = [x.decode('utf-8') for x in interm]
        return interm