"""
    Used for data sources where there is only one text domain. 
    Eg not suitable for transformer models where there is both a context and content sections
"""

import typing

from src.data.DataObjects.Text.TextDataObjectBase import TextDataObjectBase, ObjectHistoryContainer
from src.data.Preprocessing.Text.Tokenise.BaseClass import TokenBaseClass


class SingleDomainDataObject:
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
        self.final_train = self.train_data_hist.label_data.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        self.final_val = self.val_data_hist.label_data.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.final_train, self.final_val
