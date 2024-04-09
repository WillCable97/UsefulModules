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


class TextDataobjectAutoregressorGeneral:
    def run_both_processing_pipelines(self, training_data: str, val_data: str, **kwargs):
        train_history = self.run_single_processing_pipeline(training_data, **kwargs)
        val_history = self.run_single_processing_pipeline(val_data, **kwargs)
        return train_history, val_history

    def run_single_processing_pipeline(self, raw_data_source:str, text_sequencer: TokenBaseClass
                                , input_parser: typing.Callable[[str], list], label_method: typing.Callable
                                , parser_kwargs = {}) -> ObjectHistoryContainer:
        parsed_data = input_parser(raw_data_source, **parser_kwargs)
        token_text = text_sequencer.tokenise(parsed_data)
        label_data = token_text.map(label_method)
        return_obj = ObjectHistoryContainer(raw_data=raw_data_source, parsed_data=parsed_data, token_text=token_text, label_data=label_data)
        return return_obj

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        self.final_train = self.labeled_train.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        self.final_val = self.labeled_val.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.final_train, self.final_val




class SingleDomainDataObject(TextDataobjectAutoregressorGeneral):
    def __init__(self, text_sequencer: TokenBaseClass
                 ,input_reader: typing.Callable[[typing.Any], list]
                 ,input_parser: typing.Callable[[str], list]
                 ,label_method: typing.Callable
                 ,data_read_kwargs = {}, parser_kwargs = {}):
        self.text_sequencer = text_sequencer
        raw_data, training_data, val_data = input_reader(**data_read_kwargs)
        self.text_sequencer.generate_vocab(raw_data)
        self.train_data_hist, self.val_data_hist = self.run_both_processing_pipelines(training_data=training_data, val_data=val_data
                                                                                      , text_sequencer=text_sequencer, input_parser=input_parser, label_method=label_method
                                                                                      , parser_kwargs=parser_kwargs)
        self.labeled_train = self.train_data_hist.label_data
        self.labeled_val = self.val_data_hist.label_data



class MultiDomainDataObject(TextDataobjectAutoregressorGeneral):
    def __init__(self, text_sequencers: list[TokenBaseClass]
                 ,input_readers: list[typing.Callable[[typing.Any], list]]
                 ,input_parsers: list[typing.Callable[[str], list]]
                 ,label_methods: list[typing.Callable]
                 ,domain_merge_method = typing.Callable
                 ,data_read_kwargs = [{}], parser_kwargs = [{}]):
        
        self.text_sequencers = text_sequencers
        domain_count = len(text_sequencers)
        aggregated_histories = []

        for i in range(domain_count):
            #tokeniser_used = self.text_sequencers[i]
            raw_data, training_data, val_data = input_readers[i](**data_read_kwargs[i])
            self.text_sequencers[i].generate_vocab(raw_data)
            
            train_data_hist, val_data_hist = self.run_both_processing_pipelines(training_data=training_data, val_data=val_data
                                                                                        , text_sequencer=text_sequencers[i], input_parser=input_parsers[i], label_method=label_methods[i]
                                                                                        , parser_kwargs=parser_kwargs[i])
            aggregated_histories.append([train_data_hist, val_data_hist])
        self.aggregated_histories = aggregated_histories
        self.labeled_train, self.labeled_val = domain_merge_method(*aggregated_histories)

