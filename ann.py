import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("processed_heart.csv")

# Select features (X) and target (y)
X = dataset.iloc[:, :-1]  # All columns except the last one
y = dataset.iloc[:, -1]   # The last column (target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)  # Use transform on test data, not fit_transform

# Build neural network model
model = Sequential()

# Input layer and 1st hidden layer
model.add(Dense(units=145, activation="relu", input_dim=X_train.shape[1]))  # Input layer: 13 features

# Additional hidden layers
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=70, activation="relu"))

# Output layer (binary classification)
model.add(Dense(units=1, activation="sigmoid"))

# Display model summary
model.summary()

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model_his = model.fit(X_train, y_train, validation_split=0.30, batch_size=55, epochs=25, verbose=1)

# Predict using the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.45)  # Convert probabilities to binary outcomes

# Evaluate the model's performance
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot accuracy history
plt.plot(model_his.history['accuracy'])
plt.plot(model_his.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss history
plt.plot(model_his.history['loss'])
plt.plot(model_his.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save the trained model
model.save('hdp.keras')

