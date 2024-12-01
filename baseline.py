import pandas as pd

# Load the dataset
data = pd.read_csv("winequality-red.csv")

# Find the most frequent class (mode) in the target column
most_frequent_class = data['quality'].mode()[0]
print(f"Most frequent class: {most_frequent_class}")
# Create baseline predictions
baseline_predictions = [most_frequent_class] * len(data)

# Display the first few baseline predictions
print(baseline_predictions[:10])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Separate features and target variable
X = data.drop("quality", axis=1)
y = data["quality"]

# Split into training and test sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Generate baseline predictions for the test set
baseline_predictions = [most_frequent_class] * len(y_test)

# Evaluate baseline model
accuracy = accuracy_score(y_test, baseline_predictions)
precision = precision_score(y_test, baseline_predictions, average="weighted")
recall = recall_score(y_test, baseline_predictions, average="weighted")

print(f"Baseline Accuracy: {accuracy:.2f}")
print(f"Baseline Precision: {precision:.2f}")
print(f"Baseline Recall: {recall:.2f}")
