import pandas as pd

# Load the dataset
data = pd.read_csv("winequality-red.csv")

# Display basic information
print(data.head())
# Check dataset shape
print(f"Dataset shape: {data.shape}")

# Check for missing values
print(data.isnull().sum())

# Summarize the dataset
print(data.describe())
data.fillna(data.mean(), inplace=True)

from sklearn.preprocessing import StandardScaler

# Separate features and labels
X = data.drop("quality", axis=1)  # Features
y = data["quality"]              # Labels

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split into training (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check sizes
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


import seaborn as sns
import matplotlib.pyplot as plt

# Plot feature distributions
for col in data.columns[:-1]:  # Exclude 'quality'
    sns.histplot(data[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()
