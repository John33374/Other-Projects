import numpy as np

# Toy data: 4 samples, 4 features each (2x2 image flattened)
X = np.array([
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 0, 1, 1]
])
y = np.array([0, 0, 1, 1])

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    print("Initialized shapes:")
    print(" W1:", W1.shape, "b1:", b1.shape)
    print(" W2:", W2.shape, "b2:", b2.shape)
    return W1, b1, W2, b2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / exp_z.sum(axis=0, keepdims=True)

def forward(X, W1, b1, W2, b2):
    print("\n[Forward pass]")
    print(" Input X:", X.shape)
    Z1 = W1 @ X + b1
    print(" Z1:", Z1.shape)
    A1 = sigmoid(Z1)
    print(" A1:", A1.shape)
    Z2 = W2 @ A1 + b2
    print(" Z2:", Z2.shape)
    A2 = softmax(Z2)
    print(" A2:", A2.shape)
    return Z1, A1, Z2, A2

def compute_loss(A2, y):
    m = y.size
    p_correct = A2[y, np.arange(m)]
    loss = np.mean(-np.log(p_correct + 1e-12))
    print(" Loss:", loss)
    return loss

def backward(X, Y_onehot, Z1, A1, A2, W2):
    print("\n[Backward pass]")
    m = X.shape[1]
    dZ2 = A2 - Y_onehot
    print(" dZ2:", dZ2.shape)
    dW2 = (1/m) * dZ2 @ A1.T
    print(" dW2:", dW2.shape)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    print(" db2:", db2.shape)
    dA1 = W2.T @ dZ2
    print(" dA1:", dA1.shape)
    dZ1 = dA1 * A1 * (1 - A1)
    print(" dZ1:", dZ1.shape)
    dW1 = (1/m) * dZ1 @ X.T
    print(" dW1:", dW1.shape)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    print(" db1:", db1.shape)
    return dW1, db1, dW2, db2

def one_hot(y, num_classes):
    one_hot_y = np.zeros((num_classes, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    print("One-hot Y:", one_hot_y.shape)
    return one_hot_y

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=0)

# Prepare data with samples as columns
X_train = X.T
print("X_train:", X_train.shape)
Y_train = one_hot(y, 2)

# Init
W1, b1, W2, b2 = initialize_parameters(4, 4, 2)

# Train (only 3 steps so we can read the printouts)
lr = 0.1
for i in range(3):
    Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)
    loss = compute_loss(A2, y)
    dW1, db1, dW2, db2 = backward(X_train, Y_train, Z1, A1, A2, W2)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2


print("Predictions:", predict(X_train, W1, b1, W2, b2))
print("Labels:     ", y)

