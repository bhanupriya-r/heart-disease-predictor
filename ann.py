import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("processed_heart.csv")

# Select features (X) and target (y)
X = dataset.iloc[:, :-1]  # All columns except the last one
y = dataset.iloc[:, -1]   # The last column (target)

print("Class Distribution:\n", y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)  # Use transform on test data, not fit_transform

# Build neural network model
model = Sequential()

# Input layer and 1st hidden layer with L2 regularization and batch normalization
model.add(Dense(units=128, activation="relu", input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.3))  # Increased dropout rate

# 2nd hidden layer with L2 regularization and batch normalization
model.add(Dense(units=64, activation="relu", kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.3))  # Increased dropout rate

# 3rd hidden layer with L2 regularization and batch normalization
model.add(Dense(units=32, activation="relu", kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.3))  # Increased dropout rate

# Output layer (binary classification)
model.add(Dense(units=1, activation="sigmoid"))

# Display model summary
model.summary()

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Early stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    patience=15,         # Stop after 15 epochs if no improvement
    restore_best_weights=True  # Restore the best model weights
)

# Learning rate reduction callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.2,          # Reduce learning rate by a factor of 0.2
    patience=5,          # Wait for 5 epochs before reducing learning rate
    min_lr=0.0001        # Minimum learning rate
)

class_weights = {0: 1, 1: 1}  # Adjust class weights if there's class imbalance

# Train the model with early stopping and learning rate reduction
model_his = model.fit(
    X_train, y_train,
    validation_split=0.30,
    batch_size=32,
    epochs=200,  # Increased epochs (early stopping will halt training if needed)
    verbose=1,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights
)

# Predict using the test set
y_pred_prob = model.predict(X_test)
print("Prediction Probabilities:", y_pred_prob)  # Debug: Print raw probabilities

y_pred = (y_pred_prob > 0.6)  # Convert probabilities to binary outcomes
print("Binary Predictions:", y_pred)  # Debug: Print binary predictions

# Evaluate the model's performance
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)

# Adjust the threshold
threshold = 0.6  # Increase the threshold
y_pred = (y_pred_prob > threshold)

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
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss history
plt.plot(model_his.history['loss'])
plt.plot(model_his.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the trained model
model.save('hdp.keras')

# Save the scaler for later use in the Flask app
import joblib
joblib.dump(ss, 'scaler.pkl')