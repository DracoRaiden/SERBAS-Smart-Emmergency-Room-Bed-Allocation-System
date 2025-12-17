import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load the new, balanced data
df = pd.read_csv('new_random_synthetic_data.csv')

# 2. Define Features for Unsupervised Learning
# We use the key vital signs that drive risk
features = ['heart_rate', 'bp_systolic', 'bp_diastolic', 'spO2', 'temperature']
X_vitals = df[features]

# 3. Scale the Data (Crucial for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vitals)

# 4. Train the K-Means Model
# We'll use K=3 clusters to represent: 0: Low Risk, 1: Medium Risk, 2: High Risk
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 5. Determine which cluster represents High Risk
# We find the cluster whose centroid has the worst average vitals (e.g., lowest SpO2, highest HR/Temp)
cluster_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(cluster_centroids, columns=features)
print("K-Means Cluster Centroids (Mean Vitals for each group):")
print(centroid_df)

# Assume the cluster with the lowest average SpO2 is the 'High Risk' cluster
HIGH_RISK_CLUSTER_INDEX = centroid_df['spO2'].idxmin()

# 6. Save the Scaler and the K-Means Model
joblib.dump(scaler, 'vitals_scaler.joblib')
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(HIGH_RISK_CLUSTER_INDEX, 'high_risk_cluster_index.joblib')

print("\n✅ Unsupervised Models (Scaler, K-Means) trained and saved.")
print(f"   -> The High Risk Cluster Index is: {HIGH_RISK_CLUSTER_INDEX}")