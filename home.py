# Workplace Wellness Risk Monitor 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#-------------- Data Collection
df = pd.read_csv("employee_health_data.csv")

# --------- Preprocessing
features = ['heart_rate', 'stress_level', 'sleep_hours', 'absences_last_month']

# Impute missing values

df[features] = df[features].fillna(df[features].mean())

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

#------------ Anomaly Detection 
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = iso_forest.fit_predict(scaled_features)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1}) 

# --------- Risk Group Clustering 
kmeans = KMeans(n_clusters=3, random_state=42)
df['risk_group'] = kmeans.fit_predict(scaled_features)

# Map clusters to risk labels (I considered only based on mean stress level only)
risk_labels = df.groupby('risk_group')['stress_level'].mean().sort_values().index.to_list()
label_map = {risk_labels[0]: 'Low', risk_labels[1]: 'Medium', risk_labels[2]: 'High'}
df['risk_level'] = df['risk_group'].map(label_map)

# _ _ __ _ _ _Intervention Recommendation 
def recommend_intervention(row):
    if row['risk_level'] == 'High':
        return 'Counseling & Reduced Hours'
    elif row['risk_level'] == 'Medium':
        return 'Counseling'
    else:
        return 'Continue Monitoring'

df['intervention'] = df.apply(recommend_intervention, axis=1)

# --------- Visualization 
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='stress_level', y='sleep_hours', hue='risk_level', style='anomaly', s=100)
plt.title('Employee Risk Classification')
plt.xlabel('Stress Level')
plt.ylabel('Sleep Hours')
plt.show()

# ----------Report
print("\nFinal Report:")
print(df[['employee_id', 'risk_level', 'anomaly', 'intervention']]) 