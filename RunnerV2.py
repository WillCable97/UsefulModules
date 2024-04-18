import os
import tensorflow as tf

from src.data.Text.Tokens.CustomTokenisers import CustomCharacterToken
import src.data.Text.DataObjects as DataObjects


import src.models.RecurrentModels as RecModels

import src.helpers.Callbacks.BaseCallbacks as Callbacks
from src.helpers.Callbacks.TextOutputCallback import OutputTextCallback


"""

import tensorflow as tf

from src.data.DataObjects.Text.SpecificDataObjects import RegressiveSequenceTextData
from src.data.Preprocessing.Text.Tokenise.CustomTokenisers import CustomCharacterToken
import src.data.Preprocessing.Text.Tokenise as Tokens

from src.models.RecurrentModels.RNN_model import RNN_model

from ModelIO import create_model_save
"""
############################################################GENERAL VARIABLES#####################################
model_name = "W_P_RNN100_S1.1"

#File path values
root_dir = os.path.abspath("./")
processed_data = os.path.join(root_dir, "data", "processed")






############################################################DATA#####################################
#Data hyperparameters (Data should be set up such that these are only used for generating input file names)
data_soure = "HuggingFace"
data_sequencing_len = 100

data_source_base = os.path.join(processed_data, data_soure, f"Seq{data_sequencing_len}")

#Pre processing hyperparameters
token_seqence_length = 100
batch_size = 64
buffer_size = 10000

tokeniser = CustomCharacterToken(token_seqence_length)
my_data_object = DataObjects.SingleDomainDataSet(base_path=data_source_base, domain_tokeniser=tokeniser)
final_train, final_val = my_data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)


vocab_size = my_data_object.vocab_size
print(f"Data generated with vocab size: {vocab_size}")




############################################################MODELS#####################################
#Model hyperparameters
embedding_dimension = 128
dense_dimension = 512
epoch_count = 10

sequential_model = RecModels.RNN_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension
                                       , rnn_units=dense_dimension, batch_size=batch_size)


sequential_model_save = RecModels.RNN_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension
                                            , rnn_units=dense_dimension, batch_size=1)


#Compile

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
sequential_model.compile("adam", loss = loss_obj, metrics="accuracy")



#Fit
my_checkpoint_callback = Callbacks.checkpoint_callback(root_dir, model_name,5)
my_csv_callback = Callbacks.csv_callback(root_dir, model_name)
my_text_callback = OutputTextCallback(root_dir, model_name, sequential_model_save, my_data_object.tokeniser, ["the man went"], [my_data_object.tokeniser], [12]) #Hate this


sequential_model.fit(final_train, epochs=epoch_count, validation_data=final_val, callbacks=[my_text_callback])









"""

print(final_train)


#for i in final_train:
#    print(i)
#    break









from src.models.Callbacks.callbacks import checkpoint_callback, csv_callback
from src.models.Callbacks.TextOutputCallback import OutputTextCallback
my_checkpoint_callback = checkpoint_callback(root_dir, model_name,5)
my_csv_callback = csv_callback(root_dir, model_name)
my_text_callback = OutputTextCallback(root_dir, model_name, sequential_for_save, data_object.text_sequencer, ["the man went"], [data_object.text_sequencer])


sequential_inst.fit(final_train, epochs=epoch_count, validation_data=final_val, callbacks=[my_checkpoint_callback, my_csv_callback, my_text_callback])"""
"""
#Model IO
train_data = data_object.train_data_hist.label_data
val_data = data_object.val_data_hist.label_data
tokeniser = data_object.text_sequencer




sequential_for_save.build(tf.TensorShape([1, None])) 
sequential_for_save.set_weights(sequential_inst.get_weights())


create_model_save(model_name, sequential_for_save, train_data, val_data, {"primary_token": tokeniser})


"""


