import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load the trained model
# ------------------------------
model = np.load("mnist_model.npz")
W1, B1 = model['W1'], model['B1']
W2, B2 = model['W2'], model['B2']

# ------------------------------
# 2. Define activation functions
# ------------------------------
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1 @ X + B1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + B2
    A2 = softmax(Z2)
    return A2

def get_prediction(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

# ------------------------------
# 3. Load and evaluate test set
# ------------------------------
data = pd.read_csv("train.csv").values
data_test = data[:1000].T
X_test = data_test[1:] / 255.0
Y_test = data_test[0].astype(int)

A2_test = forward_prop(W1, B1, W2, B2, X_test)
predictions_test = get_prediction(A2_test)
test_acc = get_accuracy(predictions_test, Y_test)

print(f"Test set accuracy: {test_acc:.4f}")

# ------------------------------
# 4. Load and predict your image
# ------------------------------
def preprocess_custom_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # If background is bright, invert it to black
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Flatten to (784, 1)
    return img.reshape(784, 1)

# Path to your 28x28 PNG image (white digit on black)
custom_path = "9.png"
my_digit = preprocess_custom_image(custom_path)

# Predict
A2_custom = forward_prop(W1, B1, W2, B2, my_digit)
prediction = get_prediction(A2_custom)

print("Prediction for your image:", prediction[0])

# Debug: Show your image
plt.imshow(my_digit.reshape(28, 28), cmap='gray')
plt.title("Your Custom Image")
plt.axis('off')
plt.show()



# # ------------------------------
# # 5. Compare with test sample
# # ------------------------------
# for i in range(0,1000,10):
#     sample_index = i
#     sample_img = X_test[:, sample_index].reshape(784, 1)
#     sample_label = Y_test[sample_index]

#     plt.imshow(sample_img.reshape(28, 28), cmap='gray')
#     plt.title("MNIST Test Sample")
#     plt.axis('off')
#     plt.show()

#     A2_sample = forward_prop(W1, B1, W2, B2, sample_img)
#     sample_pred = get_prediction(A2_sample)[0]

#     print(f"Test sample prediction: {sample_pred}, True label: {sample_label}")

    