import numpy as np
from exam import hours_studied, calculated_coefficients, intercept

print(intercept)

def log_odds(features, coefficients,intercept):
  return np.dot(features,coefficients) + intercept

def sigmoid(z):
    denominator = 1 + np.exp(-z)
    return 1/denominator

# Create predict_class() function here

def predict_class(features, coefficients, intercept, threshold):
  calculated_log_odds = log_odds(features, coefficients, intercept)
  probabilities = sigmoid(calculated_log_odds)
  return np.where(probabilities >= threshold, 1, 0)
  # array_with_threshold = []
  # for i in probabilities:
  #   if i >= threshold:
  #     array_with_threshold.append(1)
  #   elif i <= threshold:
  #     array_with_threshold.append(0)

#final Notes; MAKE SURE TO UNDERSTAND HOW CALCULATED_COEFFICIENTS, intercept are calculated. 

final_results = predict_class(hours_studied, calculated_coefficients, intercept, 0.5)

print(final_results
