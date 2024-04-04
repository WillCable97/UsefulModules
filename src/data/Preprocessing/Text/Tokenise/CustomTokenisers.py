import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.data.Preprocessing.Text.Tokenise.BaseClass import TokenBaseClass


def create_mapping_from_vocab(vocab: list):
    word_index = {c: i+1 for i,c in enumerate(vocab)}
    index_word = {word_index[c]: c for c in word_index}
    return word_index, index_word


class CustomCharacterToken(TokenBaseClass):
    """
        Tokeniser for charachter level data
    """
    def __init__(self):
        self.pad_token = 0
        self.start_string = "^" #Not implemented
        self.end_string = "^" #Not implemented

    def generate_vocab(self, input: list) -> None:
        full_text = ' '.join(input)
        vocab = sorted(set(full_text))
        if not self.start_string in vocab: vocab.append(self.start_string)
        if not self.end_string in vocab: vocab.append(self.start_string)
        self.vocab = vocab
        self.vocab_size = len(vocab)

        #objects needed to perform the tokenising
        self.word_index, self.index_word = create_mapping_from_vocab(vocab=vocab)


    def tokenise(self, input: list, sequence_length: int = None) -> tf.Tensor:
        mapped_text = [[self.word_index[char_found] for char_found in sentence] for sentence in input]        
        padded_text = pad_sequences(mapped_text, padding ="post", maxlen=sequence_length, truncating='post')
        tensor_val = tf.data.Dataset.from_tensor_slices(padded_text)
        return tensor_val

    def detokenise(self, input: tf.Tensor) -> list:
        split_list = [[self.index_word[char_found] for char_found in sentence] for sentence in input.numpy()]
        return [''.join(split_sent) for split_sent in split_list]
