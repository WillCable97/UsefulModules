import typing
import tensorflow as tf

from src.data.Preprocessing.Text.Tokenise.BaseClass import TokenBaseClass
from src.data.DataObjects.Text.TextDataObjectBase import SingleDomainDataObject, MultiDomainDataObject
import data.Text.IO as TextIO
from src.data.Preprocessing.Text.TextParsing import sequence_string
from src.data.DataObjects.Text.LabelFunctions import standard_autoregressive_label

#####################################_DESCRIPTION_#####################################
#Data reads from text file and sequences in standard autogressive way (predicting next token)
class RegressiveSequenceTextData(SingleDomainDataObject):
    def __init__(self, parent_data_file_path: str, text_sequencer: TokenBaseClass):
       super().__init__(text_sequencer, TextIO.full_text_loader, sequence_string, standard_autoregressive_label
                        , {"parent_path" : parent_data_file_path}, {"split_on" : "[SEQ_SPLITTER]", "sequence_len": 1})
       self.vocab_size = self.text_sequencer.vocab_size






#####################################_DESCRIPTION_#####################################
def reorder_transformer_dataset(context_tensor, content_tensor):
   (a, _) = context_tensor
   (c, d) = content_tensor
   return (a,c) , d #Context, Content, Offset content 

def transformer_domain_merge(context_data, content_data):
    train_set = tf.data.Dataset.zip((context_data[0].label_data, content_data[0].label_data))
    val_set = tf.data.Dataset.zip((context_data[1].label_data, content_data[1].label_data))
    return train_set.map(reorder_transformer_dataset), val_set.map(reorder_transformer_dataset)

class RegressiveBiDomainTextdata(MultiDomainDataObject):
    def __init__(self, parent_data_file_path: str, text_sequencer: TokenBaseClass, dataset_sequence_length: int):
        super().__init__([text_sequencer, text_sequencer], [TextIO.full_text_loader, TextIO.full_text_loader]
                         , [sequence_string, sequence_string], [standard_autoregressive_label, standard_autoregressive_label]
                         , transformer_domain_merge
                         , dataset_sequence_length
                         , [{"parent_path" : parent_data_file_path, "encoding":"utf8", "file_prefix": "context_"}
                            , {"parent_path" : parent_data_file_path, "encoding":"utf8", "file_prefix": "content_"}]
                         , [{"split_on" : "[SEQ_SPLITTER]", "sequence_len": 1}, {"split_on" : "[SEQ_SPLITTER]", "sequence_len": 1}])
        self.vocab_sizes = [tokeniser.vocab_size for tokeniser in self.text_sequencers]