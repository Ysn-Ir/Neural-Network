import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load and Prepare Data
# ----------------------------

data = pd.read_csv("train.csv").values
np.random.shuffle(data)

data_test = data[:1000].T
X_test = data_test[1:] / 255.0
Y_test = data_test[0].astype(int)

data_train = data[1000:].T
X = data_train[1:] / 255.0
Y = data_train[0].astype(int)

# ----------------------------
# 2. Neural Network Functions
# ----------------------------

def init():
    W1 = np.random.randn(100, 784) * 0.01
    B1 = np.zeros((100, 1))
    W2 = np.random.randn(10, 100) * 0.01
    B2 = np.zeros((10, 1))
    return W1, B1, W2, B2

def relu(Z):
    return np.maximum(0, Z)

def drelu(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def onehot(Y):
    onehot_Y = np.zeros((10, Y.size))
    onehot_Y[Y, np.arange(Y.size)] = 1
    return onehot_Y

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1 @ X + B1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    onehot_Y = onehot(Y)

    dZ2 = A2 - onehot_Y
    dW2 = (1/m) * dZ2 @ A1.T
    dB2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = (W2.T @ dZ2) * drelu(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    dB1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, dB1, dW2, dB2

def update(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    W1 -= alpha * dW1
    B1 -= alpha * dB1
    W2 -= alpha * dW2
    B2 -= alpha * dB2
    return W1, B1, W2, B2

def get_prediction(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

# ----------------------------
# 3. Train and Save the Model
# ----------------------------

def train(X, Y, iterations=1000, alpha=0.01):
    W1, B1, W2, B2 = init()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, B1, W2, B2 = update(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

        if i % 100 == 0:
            acc = get_accuracy(get_prediction(A2), Y)
            print(f"Iteration {i}, Training Accuracy: {acc:.4f}")

    return W1, B1, W2, B2

# Train the model
W1, B1, W2, B2 = train(X, Y, iterations=2000, alpha=0.1)
np.savez("mnist_model.npz", W1=W1, B1=B1, W2=W2, B2=B2)

# ----------------------------
# 4. Evaluate on Test Set
# ----------------------------

_, _, _, A2_test = forward_prop(W1, B1, W2, B2, X_test)
preds_test = get_prediction(A2_test)
print("Test Set Accuracy:", get_accuracy(preds_test, Y_test))

# ----------------------------
# 5. Predict Custom Image
# ----------------------------

def load_my_digit_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot find image: {path}")

    # If image is white background, invert it
    if np.mean(img) > 127:
        img = 255 - img

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Normalize
    img = img.astype(np.float32) / 255.0

    return img.reshape(784, 1)

def show_image(img_flat):
    plt.imshow(img_flat.reshape(28, 28), cmap='gray')
    plt.title("Custom Image")
    plt.axis('off')
    plt.show()

# Load and predict
custom_image_path = "9.png"
my_digit = load_my_digit_image(custom_image_path)
show_image(my_digit)

_, _, _, A2_custom = forward_prop(W1, B1, W2, B2, my_digit)
prediction = get_prediction(A2_custom)
print("Prediction for custom image:", prediction[0])
