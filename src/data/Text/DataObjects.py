from src.data.Text.Tokens.BaseClass import TokenBaseClass
from src.data.Text.TextProcessor import TextProcessor
import src.data.Text.IO as txtIO

import tensorflow as tf

class GeneralTextDataObject:
    def __init__(self):
        self.final_train = None
        self.final_validation = None

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        self.batched_train = self.final_train.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        self.batched_validation = self.final_validation.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.batched_train, self.batched_validation






#############DEFINED FOR AUTOREGRESSIVE MODELS####################
class SingleDomainDataSet(GeneralTextDataObject):
    def __init__(self, base_path: str, domain_tokeniser: TokenBaseClass):
        super().__init__()
        self.tokeniser = domain_tokeniser

        base_path, train_path, validation_path = txtIO.base_train_val_paths(base_path)
        base_processor = TextProcessor()
        base_processor.read_text_from_file(base_path)
        self.tokeniser.generate_vocab([base_processor.raw_data])

        self.vocab_size = self.tokeniser.vocab_size

        self.processed_train, self.final_train = self.process_text(train_path, self.tokeniser)
        self.processed_validation, self.final_validation = self.process_text(validation_path, self.tokeniser)

    def process_text(self, data_file_path:str, data_tokeniser: TokenBaseClass) ->TextProcessor: 
        return_processor = TextProcessor()
        return_processor.read_text_from_file(data_file_path)
        return_processor.split_text_by_char("[SEQ_SPLITTER]", keep_pivot_string=False)
        token_data =data_tokeniser.tokenise(return_processor.split_raw_data, data_tokeniser.sequence_len)
        labeled_data = token_data.map(lambda input_tensor: (input_tensor[:-1], input_tensor[1:])) #Autoregressive label
        return return_processor, labeled_data




class BiDomainDataSet(GeneralTextDataObject):
    def __init__(self, context_base_path: str, content_base_path: str
                 , context_tokeniser, content_tokeniser):
        super().__init__()
        self.conext_domain = SingleDomainDataSet(base_path=context_base_path, domain_tokeniser=context_tokeniser)
        self.content_domain = SingleDomainDataSet(base_path=content_base_path, domain_tokeniser=content_tokeniser)

        #Training data
        context_train = self.conext_domain.final_train
        content_train = self.content_domain.final_train
        merged_train = tf.data.Dataset.zip((context_train, content_train))

        #Validation data
        context_validation = self.conext_domain.final_validation
        content_validation = self.content_domain.final_validation
        merged_validation = tf.data.Dataset.zip((context_validation, content_validation))

        self.final_train = merged_train.map(self.create_merged_set)
        self.final_validation = merged_validation.map(self.create_merged_set)

    def create_merged_set(self, context, content):
        context_feat, _ = context
        content_feat, content_label = content
        return (context_feat, content_feat), content_label