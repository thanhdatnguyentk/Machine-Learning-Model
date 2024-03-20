import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# Generate random data
X = np.random.rand(1000, 1)
y = 5 + 3 * X + 0.5 * np.random.randn(1000, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the trained model
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error by using Linear Regression:", mse)

# Define gradient descent functions
def grad(w, X, y):
    N = X.shape[0]
    Xbar = np.concatenate((np.ones((N, 1)), X), axis=1)
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w, X, y):
    N = X.shape[0]
    Xbar = np.concatenate((np.ones((N, 1)), X), axis=1)
    return 0.5/N * np.linalg.norm(y - Xbar.dot(w))**2

def myMiniBatchGD(w_init, eta, X, y, batch_size, num_epochs):
    w = [w_init]
    N = X.shape[0]
    num_batches = N // batch_size
    for epoch in range(num_epochs):
        shuffled_indices = np.random.permutation(N)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            w_new = w[-1] - eta * grad(w[-1], X_batch, y_batch)
            if np.linalg.norm(grad(w_new, X_batch, y_batch)) / len(w_new) < 1e-3:
                break
            w.append(w_new)
    return (w, epoch)

# Perform gradient descent on training data
w_init = np.array([[2], [1]])
(w_train, it_train) = myMiniBatchGD(w_init, 1, X_train, y_train, batch_size=32, num_epochs=100)
y_pred_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1).dot(w_train[-1])
mse_train = mean_squared_error(y_train, y_pred_train)
print("Mean Squared Error on Training Data:", mse_train)
print('Solution found by GD: w =', w_train[-1].T, ', after', it_train+1, 'iterations.')

# Perform gradient descent on test data
y_pred_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1).dot(w_train[-1])
mse_test = mean_squared_error(y_test, y_pred_test)
print("Mean Squared Error on Test Data:", mse_test)

# Plot the scatter plot and regression line
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted by Linear Regression')
plt.plot(X_test, y_pred_test, color='green', label='Predicted by Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter plot of data')
plt.legend()

# Save the plot
plt.savefig('./Gradient-Descent/plot.png')