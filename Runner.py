import os
import tensorflow as tf

from src.data.DataObjects.Text.SingleDomainDataObjects.SpecificImplementations import RegressiveSequenceTextData
from src.data.Preprocessing.Text.Tokenise.CustomTokenisers import CustomCharacterToken
import src.data.Preprocessing.Text.Tokenise as Tokens

from src.models.RecurrentModels.RNN_model import RNN_model

from ModelIO import create_model_save

############################################################HYPER PARAMTERS#####################################

#Meta Info
model_name = "W_P_RNN100_S1.1"

#Data hyperparameters
data_soure = "HuggingFace"
data_sequencing_len = 100

#Pre processing hyperparameters
token_seqence_length = 100
batch_size = 64
buffer_size = 10000

#Model hyperparameters
embedding_dimension = 128
dense_dimension = 512
epoch_count = 20

#File path values
root_dir = os.path.abspath("./")
processed_data = os.path.join(root_dir, "data", "processed")
data_source_base = os.path.join(processed_data, data_soure, f"Seq{data_sequencing_len}")

#Data objects
tokeniser = Tokens.CustomTokenisers.CustomCharacterToken()
data_object = RegressiveSequenceTextData(parent_data_file_path = data_source_base, text_sequencer = tokeniser, split_on = "char", sequence_len=data_sequencing_len)

vocab_size = data_object.vocab_size
print(vocab_size)

#Model
sequential_inst = RNN_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension, rnn_units=dense_dimension
                            ,batch_size=batch_size)

#Compile
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
sequential_inst.compile("adam", loss = loss_obj, metrics="accuracy")

#Train
final_train, final_val = data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)



from src.models.Callbacks.callbacks import checkpoint_callback
my_checkpoint_callback = checkpoint_callback(root_dir, model_name,5)
sequential_inst.fit(final_train, epochs=epoch_count, validation_data=final_val, callbacks=[my_checkpoint_callback])

#Model IO
train_data = data_object.train_data_hist.label_data
val_data = data_object.val_data_hist.label_data
tokeniser = data_object.text_sequencer


sequential_for_save = RNN_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension, rnn_units=dense_dimension
                            ,batch_size=1)

sequential_for_save.build(tf.TensorShape([1, None])) 
sequential_for_save.set_weights(sequential_inst.get_weights())


create_model_save(model_name, sequential_for_save, train_data, val_data, {"primary_token": tokeniser})





