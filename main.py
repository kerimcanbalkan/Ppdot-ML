import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your data
data = pd.read_csv("data/data1.csv")

# Separate features (P0, P1) and labels (Type)
X = data[["P0", "P1", "BSURF"]]
y = data["TYPE"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Fit the model on the training data
clf.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_scaled)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display additional metrics
print(classification_report(y_test, y_pred))


# Function to plot data with logarithmic scales and consistent colors for each type
def plot_data(X, y, title, colors=None):
    types = np.unique(y)
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(types)))

    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel("Period Derivative")

    for t, color in zip(types, colors):
        indices = y == t
        plt.scatter(
            np.log10(X.loc[indices, "P0"]),
            np.log10(X.loc[indices, "P1"]),
            label=f"{t}",
            color=color,
        )

    plt.legend()
    plt.show()
    plt.savefig(f"{title}.png")


# Plot the original data
plot_data(X, y, "Original Data")

# Plot the test data
plot_data(
    X_test,
    y_test,
    "Test Data",
    colors=plt.cm.rainbow(np.linspace(0, 1, len(np.unique(y)))),
)

# Plot the predicted data
plot_data(
    X_test,
    y_pred,
    "Predicted Data",
    colors=plt.cm.rainbow(np.linspace(0, 1, len(np.unique(y)))),
)
