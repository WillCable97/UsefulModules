import os
import tensorflow as tf

from src.data.Text.Tokens.CustomTokenisers import CustomCharacterToken
import src.data.Text.DataObjects as DataObjects


import src.models.RecurrentModels as RecModels

import src.helpers.Callbacks.BaseCallbacks as Callbacks
#from src.helpers.Callbacks.TextOutputCallback import OutputTextCallback

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
tokeniser_used = my_data_object.tokeniser
print(f"Data generated with vocab size: {vocab_size}")




############################################################MODELS#####################################
#Model hyperparameters
embedding_dimension = 128
dense_dimension = 512

sequential_model = RecModels.RNN_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension
                                       , rnn_units=dense_dimension, batch_size=1)








import src.helpers.TextGenerator as Gen



#INPUT TESTS
input_handle = Gen.TextSequenceInputHandler("Hello there", tokeniser_used)
print(input_handle.create_model_input())

#input_handle = Gen.TextSequenceInputHandler("Hello there", tokeniser_used, sequential_model, "Hello there", tokeniser_used)
#print(input_handle.create_model_input())


#input_handle = Gen.TextSequenceInputHandler("Hello there", tokeniser_used, sequential_model
#                                            , ["Hello there", "Another"], [tokeniser_used, tokeniser_used])
#print(input_handle.create_model_input())


model_handle = Gen.ModelHandler(sequential_model)
new_token = model_handle.get_new_token_by_max(input_handle.create_model_input())
print(new_token)


initial_input = input_handle.auto_regressive_tensor
output_handle = Gen.OutputSequence(tokeniser_used)
output_handle.run_init(init_tokens=initial_input)
output_handle.add_token(new_token)


print(initial_input)
print(output_handle.create_new_input())



test_generator = Gen.TextGenerator("Hello there", tokeniser_used, sequential_model)
test_generator.generate_sequence(50)

