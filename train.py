import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.theta = np.array([0.0, 0.0])
        self.data = None

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        self.original_features = np.array(self.data.iloc[:, 0])
        self.original_target = np.array(self.data.iloc[:, 1])
        self.normalize_data()

    def save_thetas(self):
      with open('model.42', 'w') as file:
        file.write(str(self.theta[0]) + "," + str(self.theta[1]))

    def normalize_data(self):
        self.min_features = np.min(self.original_features)
        self.max_features = np.max(self.original_features)
        self.min_target = np.min(self.original_target)
        self.max_target = np.max(self.original_target)

        self.features = (self.original_features - self.min_features) / (self.max_features - self.min_features)
        self.target = (self.original_target - self.min_target) / (self.max_target - self.min_target)

    def denormalize_theta(self):
        self.theta[1] = (self.max_target - self.min_target) * self.theta[1] / (self.max_features - self.min_features)
        self.theta[0] = self.min_target + ((self.max_target - self.min_target) * self.theta[0]) + self.theta[1] * (1 - self.min_features)

    def predict(self, features):
        return self.theta[0] + (self.theta[1] * features)

    def mean_squared_error(self):
        predictions = self.predict(self.features)
        return np.mean((predictions - self.target) ** 2)

    def gradient(self):
        errors = self.predict(self.features) - self.target
        gradient_1 = np.mean(errors * self.features)
        gradient_0 = np.mean(errors)
        return np.array([gradient_0, gradient_1])

    def gradient_descent(self, tolerance=1e-7, max_iterations=10000):
        prev_mse = 0
        for _ in range(max_iterations):
            gradient = self.gradient()
            self.theta -= self.learning_rate * gradient

            cur_mse = self.mean_squared_error()
            if abs(cur_mse - prev_mse) < tolerance:
                break

            prev_mse = cur_mse

        self.denormalize_theta()
        self.save_thetas()

    def plot(self):
        plt.scatter(self.original_features, self.original_target)
        print("Theta 0 =", self.theta[0], ", Theta 1 =", self.theta[1])
        x_values = np.array([min(self.original_features), max(self.original_features)])
        y_values = self.predict(x_values)
        plt.plot(x_values, y_values, color='red')
        plt.show()

filepath = ""
while True:
  try:
      filepath = str(input("Enter the dataset path : "))
      break
  except ValueError:
      print("Error: not a path")


linear = LinearRegression(0.01)
try:
  linear.load_data(filepath)
  linear.gradient_descent()
  linear.plot()
except:
  print("Error: file not found")
