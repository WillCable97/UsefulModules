import os
import tensorflow as tf

from src.data.DataObjects.Text.SpecificDataObjects import RegressiveBiDomainTextdata
from src.data.Preprocessing.Text.Tokenise.TensorflowTokenisers import TensorflowWordTokeniser
from src.data.Preprocessing.Text.Tokenise.CustomTokenisers import CustomCharacterToken
import src.data.Preprocessing.Text.Tokenise as Tokens

#from src.models.RecurrentModels.RNN_model import RNN_model
from src.models.Transformer.Transformer import Transformer

from helpers.ModelIO import create_model_save

############################################################HYPER PARAMTERS#####################################

#Meta Info
model_name = "W_P_Trans100_S1.1"

#Data hyperparameters (Data should be set up such that these are only used for generating input file names)
data_soure = "Webscrape"
data_sequencing_len = 1

#Pre processing hyperparameters
token_seqence_length = 100
batch_size = 64
buffer_size = 10000

#Model hyperparameters
embedding_dimension = 64
dense_dimension = 64
num_heads = 2
num_att_layers = 2
dropout_rate = 0.1
epoch_count = 15


#File path values
root_dir = os.path.abspath("./")
processed_data = os.path.join(root_dir, "data", "processed")
data_source_base = os.path.join(processed_data, data_soure, f"Seq{data_sequencing_len}")

#Data objects
tokeniser = CustomCharacterToken()
data_object = RegressiveBiDomainTextdata(parent_data_file_path=data_source_base, text_sequencer=tokeniser, dataset_sequence_length=token_seqence_length)


for i in data_object.aggregated_histories:
    for j in i:
        print(j.label_data)


print(f"{data_object.labeled_train} \n {data_object.labeled_val}")





context_vocab = data_object.vocab_sizes[0]
content_vocab = data_object.vocab_sizes[1]
print(f"Context Vocab: {context_vocab}, Content Vocab: {content_vocab}")


#Model
transformer_model = Transformer(vocab_size=content_vocab, context_vocab_size=context_vocab, embedding_dimension=embedding_dimension
                                , context_length=token_seqence_length-1, content_length=token_seqence_length-1, num_heads=num_heads
                                , dense_dimension=dense_dimension, num_att_layers=num_att_layers) #Token length need to be revised 

#Compile
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
transformer_model.compile("adam", loss = loss_obj, metrics="accuracy")

#Train
final_train, final_val = data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)

print(final_train)
print(final_val)



for i in final_train: 
    inputs, output = i
    print(inputs)
    print(output)
    break


from src.models.Callbacks.callbacks import checkpoint_callback, csv_callback
from src.models.Callbacks.TextOutputCallback import OutputTextCallback
my_checkpoint_callback = checkpoint_callback(root_dir, model_name,5)
my_csv_callback = csv_callback(root_dir, model_name)
my_text_callback = OutputTextCallback(root_dir, model_name, transformer_model, data_object.text_sequencers[1]
                                      , ["the man went", "*"], data_object.text_sequencers, [99,99])




transformer_model.fit(final_train, epochs=epoch_count, validation_data=final_val, callbacks=[my_checkpoint_callback, my_csv_callback, my_text_callback])
"""


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


"""


