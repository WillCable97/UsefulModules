import tensorflow as tf
from ModelIO import ModelFromFile


model_name = "W_P_RNN100_S1.0"


model_loader = ModelFromFile(model_name)
model_loader.perform_load()
loaded_token = model_loader.tokenisers['primary_token']
print(loaded_token)

for i in model_loader.validation_data:
    inputs = tf.expand_dims(i[0], axis=0)
    print(loaded_token.detokenise(inputs)[0])
    output = model_loader.model_obj(inputs)
    print(output)
    #print(loaded_token.detokenise(output))
    #print()
    break



"""
sequential_gen_inst = RNN_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension, rnn_units=dense_dimension
                                ,batch_size=1)

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
sequential_inst.compile("adam", loss=loss_obj, metrics="accuracy")
sequential_gen_inst.build(tf.TensorShape([1, None])) """