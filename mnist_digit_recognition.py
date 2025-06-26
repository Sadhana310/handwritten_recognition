import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load MNIST Dataset from OpenML
print("Loading MNIST dataset... (This may take a moment on first run)")
# scikit-learn's fetch_openml is used to avoid the TensorFlow dependency.
x, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')

# The dataset is already flattened (784 features) and contains 70,000 images.
# Standard split is 60,000 for training and 10,000 for testing.
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

# 2. Normalize Data
# Scale pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Build and Train MLP Model with scikit-learn
print("\nTraining model with scikit-learn...")
# This model has one hidden layer with 128 neurons, similar to your previous Keras model.
# `max_iter` is like epochs. Training will stop after 5 full passes over the data.
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=5, alpha=1e-4,
                    solver='adam', verbose=10, random_state=1,
                    learning_rate_init=.001)

mlp.fit(x_train, y_train)

# 4. Evaluate Model
print("\nEvaluating model...")
score = mlp.score(x_test, y_test)
print(f"\nTest Accuracy: {score * 100:.2f}%")

# 5. Full Evaluation (Optional)
print("\nClassification Report:")
y_pred = mlp.predict(x_test)
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Show a Sample Prediction
print("\nDisplaying a sample prediction...")
sample_index = 5  # Change this to test other digits
plt.imshow(x_test[sample_index].reshape(28, 28), cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.show()

predicted_digit = mlp.predict([x_test[sample_index]])[0]
prediction_probabilities = mlp.predict_proba([x_test[sample_index]])
confidence = np.max(prediction_probabilities) * 100

print(f"Recognized Digit: {predicted_digit} (Confidence: {confidence:.2f}%)")
