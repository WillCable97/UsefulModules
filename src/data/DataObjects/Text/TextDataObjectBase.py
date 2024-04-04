"""
    Base class for handling text level data
    Will oversea the full process from raw data to final batched dataset

    There are 3 moethods:
        1- run_both_processing_pipelines: handles data processing and will return data files and history for both the training set and the validation set
        2- run_single_processing_pipeline: handles individual (eg training) dataset. Will expect methods for parsing data, sequenceing and labeling
        3- batch_and_shuffle: Returns the final data sets after shuffling and batching the labeled datas


    The init function of child classes should contain raw data retrieval to be passed through the preprocessing pipeline
"""

import typing
import tensorflow as tf
from dataclasses import dataclass

from src.data.Preprocessing.Text.Tokenise.BaseClass import TokenBaseClass

@dataclass
class ObjectHistoryContainer:
    raw_data: str
    parsed_data: list
    token_text: tf.Tensor
    label_data: typing.Any

class TextDataObjectBase(typing.Protocol):
    def run_both_processing_pipelines(self, training_data: str, val_data: str, **kwargs) -> ObjectHistoryContainer:
        ...

    def run_single_processing_pipeline(self, raw_data_source:str, text_sequencer: TokenBaseClass
                                       , input_parser: typing.Callable[[str], list], label_method: typing.Callable
                                       , parser_kwargs = {}) -> ObjectHistoryContainer:
        ...
    
    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        ...
        