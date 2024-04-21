import tensorflow as tf
import numpy as np


from helpers.ModelIO import ModelFromFile

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


model_name = "W_P_RNN100_S1.1"


model_loader = ModelFromFile(model_name)
model_loader.perform_load()
loaded_token = model_loader.tokenisers['primary_token']
validation_data = model_loader.validation_data
training_data = model_loader.training_data
model = model_loader.model_obj




from src.models.TextGenerator import TextGenerator



Gen = TextGenerator(model, loaded_token)

for feat, label in validation_data:
    feat_update = tf.expand_dims(feat, axis=0)
    Gen.load_inputs_by_token([feat_update], [loaded_token])
    print(Gen.generate_text(150))
    break








"""
predictions = []
labels = []
counter = 1

"""


"""
for feature, label in validation_data:
    if (counter - 1) % 1000 == 0: print(f"Creating records for {counter} - {counter + 1000}")
    model_input = tf.expand_dims(feature, axis =0)
    model_output = model(model_input)
    output_shape = model_output.shape
    labels.extend(label)
    prediction = model_output[0]#[output_shape[1]-1]
    predictions.extend(np.argmax(prediction, axis = 1))
    counter +=1



# Calculate accuracy
accuracy = accuracy_score(labels, predictions)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(labels, predictions, average=None)
print("Precision:", precision)

# Calculate recall
recall = recall_score(labels, predictions, average=None)
print("Recall:", recall)

# Calculate F1 score
f1 = f1_score(labels, predictions, average=None)
print("F1 Score:", f1)




def calculate_perplexity(model, validation_dataset):
    
    Calculates the perplexity of an NLP model's predictions on a validation dataset.

    Args:
    - model: The trained TensorFlow model.
    - validation_dataset: A tf.data.Dataset object containing the validation data.

    Returns:
    - The perplexity of the model on the validation dataset.
    
    # Ensure the model is in evaluation mode
    #model.eval()

    # Cross-entropy loss
    cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Accumulate the losses here; we'll use this to compute the average loss
    total_loss = 0
    total_batches = 0

    for inputs, labels in validation_dataset:
        model_input = tf.expand_dims(inputs, axis =0)
        # Generate predictions
        predictions = model(model_input)

        # Compute the loss
        loss = cross_entropy_loss(labels, predictions)

        # Update total loss and batch count
        total_loss += loss.numpy()
        total_batches += 1

    # Calculate average loss across all batches
    average_loss = total_loss / total_batches

    # Calculate and return perplexity
    perplexity = tf.exp(average_loss).numpy()
    return perplexity

print(f"Perplexity: {calculate_perplexity(model,validation_data)}")


#print(validation_data[0])


"""
"""
for i in model_loader.validation_data:
    inputs = tf.expand_dims(i[0], axis=0)
    print(loaded_token.detokenise(inputs)[0])
    output = model_loader.model_obj(inputs)
    print(output)
    break
"""

