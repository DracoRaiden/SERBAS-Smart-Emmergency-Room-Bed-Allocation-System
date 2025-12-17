import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Load the new, balanced data
# 🔑 CRITICAL CHANGE: We are now loading the dataset with the 40-35-25 ratio.
df = pd.read_csv('new_random_synthetic_data.csv')

# 2. Define Features (X) and Target (y)
# These columns must match the expected input for your model
feature_cols = ['age', 'heart_rate', 'bp_systolic', 'bp_diastolic', 'spO2', 'temperature']
X = df[feature_cols]
y = df['severity']

# 3. Train the Random Forest Classifier
# Note: Since the data is now balanced (40-35-25 ratio), we can remove the 
# 'class_weight="balanced"' parameter, allowing the model to rely fully on 
# the feature boundaries you defined.
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# 4. Save the Model
# This will overwrite your previous model with the new, better-trained version.
joblib.dump(rf_classifier, 'random_forest_model.joblib')

print("✅ Random Forest Model Retrained Successfully!")
print("The model has been saved as random_forest_model.joblib, trained on the new balanced data.")