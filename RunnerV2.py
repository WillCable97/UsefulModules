import os
import tensorflow as tf

from src.data.Text.Tokens.CustomTokenisers import CustomCharacterToken
from src.data.Text.Tokens.TensorflowTokenisers import WordTokeniser
import src.data.Text.DataObjects as DataObjects


import src.models.RecurrentModels as RecModels

import src.helpers.Callbacks.BaseCallbacks as Callbacks
from src.helpers.Callbacks.TextOutputCallback import OutputTextCallback


import src.helpers.TextGenerator as Gen
from src.helpers.ModelIO import create_model_save


#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)


############################################################GENERAL VARIABLES#####################################
model_name = "W_P_RNN300_S1.0_Word"

#File path values
root_dir = os.path.abspath("./")
processed_data = os.path.join(root_dir, "data", "processed")


############################################################DATA#####################################
#Data hyperparameters (Data should be set up such that these are only used for generating input file names)
data_soure = "LargerText"
data_sequencing_len = 300

data_source_base = os.path.join(processed_data, data_soure)#, f"Seq{data_sequencing_len}")

#Pre processing hyperparameters
token_seqence_length = 150
batch_size = 64
buffer_size = 1000

#tokeniser = CustomCharacterToken(token_seqence_length)
tokeniser = WordTokeniser(token_seqence_length)
my_data_object = DataObjects.SingleDomainDataSet(base_path=data_source_base, domain_tokeniser=tokeniser)
final_train, final_val = my_data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)


vocab_size = my_data_object.vocab_size
tokeniser_used = my_data_object.tokeniser
print(f"Data generated with vocab size: {vocab_size}")




############################################################MODELS#####################################
#Model hyperparameters
embedding_dimension = 128
dense_dimension = 512
epoch_count = 30

sequential_model = RecModels.LSTM_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension
                                       , rnn_units=dense_dimension, batch_size=batch_size)


sequential_model_save = RecModels.LSTM_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension
                                            , rnn_units=dense_dimension, batch_size=1)

sequential_model_save.build(tf.TensorShape([1, None])) 
#Compile
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
sequential_model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_obj, metrics="accuracy")



#Fit
my_checkpoint_callback = Callbacks.checkpoint_callback(root_dir, model_name,5)
my_csv_callback = Callbacks.csv_callback(root_dir, model_name)
my_text_callback = OutputTextCallback(string_beginning="the man went to ", output_tokeniser=tokeniser_used, input_model=sequential_model_save
                                      , base_path=root_dir, input_model_name=model_name)


sequential_model.fit(final_train, epochs=epoch_count, validation_data=final_val, callbacks=[my_checkpoint_callback])


#########################################################POST MODEL CHECKS#####################################


sequential_model_save.set_weights(sequential_model.get_weights())
test_generator = Gen.TextGenerator("it was a sunny day and the man woke up to find that ", tokeniser_used, sequential_model_save)

selector="topn"
print(f"OUTPUT HERE: {test_generator.generate_sequence(100, selector)}")

"""
train_save = my_data_object.final_train
validation_save = my_data_object.final_validation

create_model_save(model_name=model_name, model_object=sequential_model_save
                  , training_data=train_save, validation_data=validation_save, tokenisers={'tokeniser': tokeniser_used})
"""