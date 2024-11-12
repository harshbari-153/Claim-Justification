# Enrollment No: p23ds004
# College: NIT Surat, Gujarat, India.
# Course: M. Tech in Data Science (2023-2025)
# Guide: Krupa K. Jariwala
# Final Year Dissertation
# Topic: Claim Justification



################## Constants ####################
MODEL_NAME = "classifier_5"
path_1 = "..\\2_1 generate sentiments\\embeddings_"
path_2 = "..\\2_3 generate distance\\embeddings_"
d1 = 21
d2 = 100
new_d = 8
#################################################



################ Import Libraries ###############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from fetch_vectors import get_array as g_a
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.decomposition import PCA
import joblib
#################################################



################### Get Labels ##################
def get_labels(labels):
  #label_mapping = {"true": 0, "mostly-true": 1, "half-true": 2, "mostly-false": 3, "false":4 , "pants-fire": 5}
  label_mapping = {"true": 2, "mostly-true": 2, "half-true": 2, "mostly-false": 0, "false": 1, "pants-fire": 2}
  
  return labels.map(label_mapping).to_numpy()
#################################################



############ Fetch Reauired Labels ##############
def get_required_labels(X, Y):
  indices = Y != 2
  
  X = X[indices]
  Y = Y[indices]
  
  return X, Y
#################################################



# Extract embeddings
X1 = np.vstack((g_a(path_1+"0.txt", d1), g_a(path_1+"1.txt", d1), g_a(path_1+"2.txt", d1), g_a(path_1+"3.txt", d1), g_a(path_1+"4.txt", d1)))
X2 = np.vstack((g_a(path_2+"0.txt", d2), g_a(path_2+"1.txt", d2), g_a(path_2+"2.txt", d2), g_a(path_2+"3.txt", d2), g_a(path_2+"4.txt", d2)))
X3 = np.hstack((X1, X2))
n = len(X3)
print("Embeddings Fetched")


# Apply PCA
print("Applying PCA")
pca_model = PCA(n_components=new_d)
X = pca_model.fit_transform(X3)

# Save the PCA model
joblib.dump(pca_model, "pca_" + MODEL_NAME[-1] + ".pkl")


# Load the PCA model
#loaded_pca = joblib.load("pca_" + MODEL_NAME[-1] + ".pkl")
print("PCA Applied")

labels = pd.read_json("..\dataset\politifact_factcheck_data.json", lines = True)
Y = get_labels(labels['verdict'])
print("Labels Fetched")


X, Y = get_required_labels(X, Y)


# train test split
#x_train, x_test, y_train, y_test = train_test_split(X[:,:-1], Y, test_size = 0.2, stratify = Y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y, random_state=100)
print("Train Test splitted")


# Assuming y_train is your target array
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Convert to a dictionary
class_weight_dict = {0: class_weights[1], 1: class_weights[0]}


# Initiate Model
model = keras.Sequential([
    keras.layers.Dense(8, input_shape = (x_train.shape[1], ), activation = 'relu'),
    #Dropout(0.1),
    keras.layers.Dense(4, activation = 'relu'),
    #Dropout(0.2),
    keras.layers.Dense(2, activation = 'relu'),
    #Dropout(0.1),
    #keras.layers.Dense(16, activation=keras.layers.LeakyReLU(alpha=0.1)),
    #keras.layers.Dense(8, activation=keras.layers.LeakyReLU(alpha=0.1)),
    keras.layers.Dense(1, activation = 'sigmoid')
])

# Compile Model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
#model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 50)
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 20, batch_size = 16, class_weight=class_weight_dict, callbacks=[early_stopping])

# Save the model
model.save(MODEL_NAME + ".keras")
print("\nTraining done model saved")


# Load the saved model
loaded_model = tf.keras.models.load_model(MODEL_NAME + ".keras")



# Make Predictions
predictions = loaded_model.predict(x_test)
#test_labels = np.argmax(predictions, axis = 1)
test_labels = (predictions >= 0.5).astype(int)



############# Generate Reports ##################
# Classification report
print("\nClassification Report of " + MODEL_NAME + ":")
print(classification_report(y_test, test_labels, zero_division=0))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_labels))

# Evaluate the accuracy
accuracy = accuracy_score(y_test, test_labels)
print(f"\nAccuracy on Test Data: {accuracy}")
#################################################