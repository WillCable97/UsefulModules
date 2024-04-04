import typing

from src.data.Preprocessing.Text.Tokenise.BaseClass import TokenBaseClass
from src.data.DataObjects.Text.SingleDomainDataObjects.SingleDomainDataObject import SingleDomainDataObject
import src.data.IO.Text as TextIO
from src.data.Preprocessing.Text.TextParsing import sequence_string
from src.data.DataObjects.Text.LabelFunctions import standard_autoregressive_label

#Data reads from text file and sequences in standard autogressive way (predicting next token)
class RegressiveSequenceTextData(SingleDomainDataObject):
    def __init__(self, parent_data_file_path: str, text_sequencer: TokenBaseClass, split_on: str, sequence_len: int):
       super().__init__(text_sequencer, TextIO.full_text_loader, sequence_string, standard_autoregressive_label
                        , {"parent_path" : parent_data_file_path}, {"split_on" : split_on, "sequence_len": sequence_len})
       



