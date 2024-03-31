import typing
import tensorflow as tf

from src.data.TextToToken.TextToToken import TextToToken
from src.data.DataObjects.helper_funcs import create_offset_labels


class StandardTextDataObject:
    def __init__(self, text_sequencer: TextToToken, data_loader: typing.Callable[[typing.Any], list]
                 ,sequence_lenth: int, init_tokens = True, **kwargs):
        #Work with Raw Dataset
        self.text_sequencer = text_sequencer
        self.raw_data = data_loader(**kwargs)
        if init_tokens: self.text_sequencer.init_with_input(self.raw_data)
        self.sequence_lenth = sequence_lenth

        #Work with token list
        self.token_list = self.text_sequencer.tokenise(self.raw_data)
        self.vocab_size = self.text_sequencer.get_vocab_size()

    def pad_sequences(self, padding_length: int) -> None:
        self.token_text = tf.keras.preprocessing.sequence.pad_sequences(self.token_list, maxlen=padding_length, padding="post")

    def unpad_sequance(self) -> None:
        pass

    def create_tf_dataset(self):
        self.tf_dataset = tf.data.Dataset.from_tensor_slices(self.token_text)

    def create_label(self):
        self.tf_dataset = self.tf_dataset.map(create_offset_labels)

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        self.tf_dataset = self.tf_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.tf_dataset


class E2EStandardTextObject(StandardTextDataObject):
    """
        Just implements all operations that will be standatd for RNN single embedded operations
    """
    def __init__(self, text_sequencer: TextToToken, data_loader: typing.Callable[[typing.Any], list]
                 , sequence_lenth: int, init_tokens = True, **kwargs):
        super().__init__(text_sequencer, data_loader, sequence_lenth, init_tokens = True, **kwargs)
        self.pad_sequences(self.sequence_lenth)
        self.create_tf_dataset()
        self.create_label()