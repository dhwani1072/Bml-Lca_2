
# WINE QUALITY PREDICTION USING K-MEANS CLUSTERING

# 1. Import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 2. Load Wine Quality dataset from UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(url, sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# 3. Display first five rows
print("\nFirst five rows:")
print(data.head())

# 4. Separate input features and target
X = data.drop("quality", axis=1)
y = data["quality"]

print("\nInput features:")
print(X.columns)

print("\nQuality values:")
print(y.value_counts().sort_index())

# 5. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Apply K-Means Clustering
n_clusters = y_train.nunique()

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

kmeans.fit(X_train_scaled)

# 8. Map each cluster to its majority quality label
cluster_to_quality = {}

for cluster in range(n_clusters):

    cluster_labels = y_train[
        kmeans.labels_ == cluster
    ]

    cluster_to_quality[cluster] = cluster_labels.mode().iloc[0]

print("\nCluster to Quality Mapping:")
print(cluster_to_quality)

# 9. Predict quality for test data
test_clusters = kmeans.predict(X_test_scaled)

y_pred = np.array([
    cluster_to_quality[cluster]
    for cluster in test_clusters
])

# 10. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n--------------------------------")
print("K-MEANS CLUSTERING RESULTS")
print("--------------------------------")

print("Number of clusters:", n_clusters)

print("Accuracy:", accuracy)

print("Accuracy Percentage:",
      round(accuracy * 100, 2), "%")

# 11. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))
