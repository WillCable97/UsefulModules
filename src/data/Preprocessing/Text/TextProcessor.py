"""
Handles reading data and formatting in required way
Should read raw data, tokenise
come equipped with useful function for doing all that
"""

import src.data.IO.Text as txtIO
import src.data.Preprocessing.Text.TextParsing as Parsing


from src.data.Preprocessing.Text.Tokenise.BaseClass import TokenBaseClass
import tensorflow as tf
from tensorflow import Tensor

class TextProcessor:
    def __init__(self):
        self.raw_data: str = None
        self.split_raw_data: list = None
        #self.tokenised_text: Tensor = None #This is actually a tf tensor

        #self.tokeniser: TokenBaseClass = None
        #self.vocab_size: int = None

    #Importing Raw data (str)
    def read_text_from_file(self, file_path: str, encoding:str = None):
        self.raw_data = txtIO.read_text_file(file_path=file_path, encoding=encoding)
    
    #Parsing Raw data ([str1, str2, ...., strn])
    def split_text_by_length(self, split_on: str, sequence_len: int, discard_overflow = True):
        self.split_raw_data = Parsing.sequence_string(input_string=self.raw_data, split_on=split_on
                                                      , sequence_len=sequence_len, discard_overflow=discard_overflow)
        
    def split_text_by_char(self, split_on: str = "\n", keep_pivot_string = True):
        self.split_raw_data = Parsing.pivot_text(input_string=self.raw_data, split_on=split_on, keep_pivot_string=keep_pivot_string)



class GeneralTextDataObject:
    def __init__(self):
        self.final_train = None
        self.final_validation = None

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        self.batched_train = self.final_train.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        self.batched_validation = self.final_validation.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.batched_train, self.batched_validation






################SPECIFIC
class SingleDomainDataSet(GeneralTextDataObject):
    def __init__(self, base_path: str, domain_tokeniser: TokenBaseClass):
        super().__init__()
        self.tokeniser = domain_tokeniser

        base_path, train_path, validation_path = txtIO.full_text_loader(base_path)
        base_processor = TextProcessor()
        base_processor.read_text_from_file(base_path)
        self.tokeniser.generate_vocab(base_processor)

        self.processed_train, self.final_train = self.process_text(train_path, self.tokeniser)
        self.processed_validation, self.final_validation = self.process_text(validation_path, self.tokeniser)

    def process_text(self, data_file_path:str, data_tokeniser: TokenBaseClass) ->TextProcessor: 
        return_processor = TextProcessor()
        return_processor.read_text_from_file(data_file_path)
        return_processor.split_text_by_char("[SEQ_SPLITTER]")
        token_data =data_tokeniser.tokenise(return_processor.split_raw_data)
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