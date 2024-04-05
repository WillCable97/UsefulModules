import os
import tensorflow as tf

from src.data.DataObjects.Text.SingleDomainDataObjects.SpecificImplementations import RegressiveSequenceTextData
from src.data.Preprocessing.Text.Tokenise.CustomTokenisers import CustomCharacterToken
import src.data.Preprocessing.Text.Tokenise as Tokens

from src.models.RecurrentModels.RNN_model import RNN_model

############################################################HYPER PARAMTERS#####################################

#Meta Info
model_name = "W_P_RNN100_S1.0"

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
epoch_count = 1

#File path values
root_dir = os.path.abspath("./")
processed_data = os.path.join(root_dir, "data", "processed")
data_source_base = os.path.join(processed_data, data_soure, f"Seq{data_sequencing_len}")

#Data objects
tokeniser = Tokens.CustomTokenisers.CustomCharacterToken()
data_object = RegressiveSequenceTextData(parent_data_file_path = "./", text_sequencer = tokeniser, split_on = "char", sequence_len=20)

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
#sequential_inst.fit(final_train, validation_data=final_val, epochs=epoch_count)

#Save model, data and token
#import tensorflow as tf

# Create a sample dataset
#dataset = tf.data.Dataset.range(10)
#dataset = dataset.batch(2)  # Batch size 2 for demonstration

# Define the file path where you want to save the dataset
#file_path = 'batched_dataset.tfrecord'

# Save the dataset

dataset = data_object.train_data_hist.label_data
tf.data.experimental.save(dataset, "./")


#self.train_data_hist.label_data
#self.val_data_hist.label_data



