import numpy as np

class Predict:
  def __init__(self):
    self.theta = np.array([0.0, 0.0])
    self.__read_thetas()

  def __read_thetas(self):
    try:
      with open('model.42', 'r') as file:
        data = file.read().split(',')
        self.theta = (float(data[0]), float(data[1]))
    except:
        self.theta = [0, 0]

  def estimatePrice(self, mileage):
    return self.theta[0] + (self.theta[1] * mileage)

mileage = 0
while True:
  try:
      mileage = int(input("Enter the mileage : "))
      predictor = Predict()
      print(predictor.estimatePrice(mileage))
      break
  except ValueError:
      print("Error: not a number")