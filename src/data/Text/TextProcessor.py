"""
Handles reading data and formatting in required way
Should read raw data, tokenise
come equipped with useful function for doing all that
"""

import tensorflow as tf
from tensorflow import Tensor

import src.data.Text.IO as txtIO
import src.data.Text.TextParsing as Parsing

class TextProcessor:
    def __init__(self):
        self.raw_data: str = None
        self.split_raw_data: list = None

    #Importing Raw data (str)
    def read_text_from_file(self, file_path: str, encoding:str = None):
        self.raw_data = txtIO.read_text_file(file_path=file_path, encoding=encoding)
    
    #Parsing Raw data ([str1, str2, ...., strn])
    def split_text_by_length(self, split_on: str, sequence_len: int, discard_overflow = True):
        self.split_raw_data = Parsing.sequence_string(input_string=self.raw_data, split_on=split_on
                                                      , sequence_len=sequence_len, discard_overflow=discard_overflow)
        
    def split_text_by_char(self, split_on: str = "\n", keep_pivot_string = True):
        self.split_raw_data = Parsing.pivot_text(input_string=self.raw_data, split_on=split_on, keep_pivot_string=keep_pivot_string)



