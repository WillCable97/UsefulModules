import tensorflow as tf
import numpy as np


from ModelIO import ModelFromFile

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


model_name = "W_P_RNN100_S1.0"


model_loader = ModelFromFile(model_name)
model_loader.perform_load()
loaded_token = model_loader.tokenisers['primary_token']
validation_data = model_loader.validation_data
training_data = model_loader.training_data
model = model_loader.model_obj


print(validation_data)
print(training_data)

predictions = []
labels = []
counter = 1

for feature, label in validation_data:
    if (counter - 1) % 1000 == 0: print(f"Creating records for {counter} - {counter + 1000}")
    model_input = tf.expand_dims(feature, axis =0)
    model_output = model(model_input)
    output_shape = model_output.shape
    labels.extend(label)
    prediction = model_output[0]#[output_shape[1]-1]
    
    #print(label.shape)
    #print(model_output.shape)
    predictions.extend(np.argmax(prediction, axis = 1))
    counter +=1

    #if counter > 1: break



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
"""
# Calculate confusion matrix
conf_matrix = confusion_matrix(labels, predictions, average=None)
print("Confusion Matrix:\n", conf_matrix)



"""






#print(validation_data[0])



"""
for i in model_loader.validation_data:
    inputs = tf.expand_dims(i[0], axis=0)
    print(loaded_token.detokenise(inputs)[0])
    output = model_loader.model_obj(inputs)
    print(output)
    break
"""

