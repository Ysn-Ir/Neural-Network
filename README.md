
---

# 🧠 MNIST Digit Classifier (NumPy Only) 95% Accuracy

This project is a simple **digit recognizer** built from scratch using **NumPy**, trained on the **MNIST** dataset (via `train.csv`), and can also predict digits from custom images (like `9.png`).

---

## 📂 Project Structure

```
📁 mnist_digit_recognizer/
├── train.csv             # Dataset (downloaded from Kaggle)
├── mnist_model.npz       # Saved model parameters
├── 9.png                 # Example custom digit image (28x28 PNG)
├── main.py               # All logic: training, testing, and prediction
└── README.md             # You're here!
```

---

## ⚙️ Requirements

Install the required libraries:

```bash
pip install numpy pandas matplotlib opencv-python pillow
```

---

## 🧱 Model Architecture

| Layer  | Size | Activation |
| ------ | ---- | ---------- |
| Input  | 784  | —          |
| Hidden | 100   | ReLU       |
| Output | 10   | Softmax    |

---

## 🏋️‍♂️ Training the Model

The model is trained using `/kaggle/input/digit-recognizer/train.csv`:

```python
W1, B1, W2, B2 = train(X, Y, iterations=2000, alpha=0.1)
np.savez("mnist_model.npz", W1=W1, B1=B1, W2=W2, B2=B2)
```

* The first **1000 rows** are used for testing.
* The rest is used for training.
* Trained parameters are saved in `mnist_model.npz`.

---

## ✅ Testing the Model

After training (or loading from file), you can evaluate the model's accuracy on the test set:

```python
model = np.load("mnist_model.npz")
W1, B1, W2, B2 = model["W1"], model["B1"], model["W2"], model["B2"]

A2_test = forward_prop(W1, B1, W2, B2, X_test)
predictions = get_prediction(A2_test)
print("Test Accuracy:", get_accuracy(predictions, Y_test))
```

---

## 🖼️ Predicting Custom Images

You can also load and predict on a 28x28 PNG image like `9.png`.

### 🔧 Image Requirements

* Size: **28x28 pixels**
* Colors: **White digit on black background**
* Grayscale image
* Preprocessing is handled for you

### 🔍 Sample Usage

```python
img = cv2.imread("9.png", cv2.IMREAD_GRAYSCALE)

if np.mean(img) > 127:
    img = 255 - img  # Invert if needed

img = img / 255.0
img = img.reshape(784, 1)

A2 = forward_prop(W1, B1, W2, B2, img)
prediction = get_prediction(A2)

print("Prediction for custom image:", prediction[0])
```

---

## 🧪 Example Output

```
Iteration 0, Training Accuracy: 0.12
Iteration 10, Training Accuracy: 0.43
...
Test Accuracy: 0.9470
Prediction for custom image: 9
```

---

## 🛠 Troubleshooting

| Issue                          | Solution                                      |
| ------------------------------ | --------------------------------------------- |
| Prediction always returns 0    | Make sure the image has white digits on black |
| Accuracy is very low           | Increase training iterations or alpha         |
| Custom image prediction is bad | Make sure it’s 28x28 and normalized to \[0,1] |

---

## 📜 License

Open source for educational purposes. Feel free to modify and expand it.

---

