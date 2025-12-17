import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Define the target counts (Total N=1200)
N_TOTAL = 1200 
TARGET_COUNTS = {
    'medium': int(N_TOTAL * 0.40), # 480 samples
    'low': int(N_TOTAL * 0.35),    # 420 samples
    'high': int(N_TOTAL * 0.25)   # 300 samples
}

# --- Function to Generate Data for a Specific Class ---
def generate_class_data(severity_label, count):
    """Generates synthetic data points that are guaranteed to fall into the specified 
       severity class based on your original triage logic (serbas.py).
    """
    
    # --- Generation Logic based on your triage rules ---
    if severity_label == 'low':
        # Must NOT meet any High or Medium criteria. Limits are strict.
        age = np.random.randint(18, 70, count) # Age < 70
        heart_rate = np.random.randint(50, 100, count) # HR <= 99
        bp_systolic = np.random.randint(95, 180, count) # SBP >= 95
        bp_diastolic = np.random.randint(60, 110, count) 
        spO2 = np.random.randint(95, 100, count) # SpO2 >= 95
        temperature = np.round(np.random.uniform(36.0, 38.4, count), 1) # Temp < 38.5

    elif severity_label == 'medium':
        # Must meet at least ONE Medium criteria and NOT any High criteria.
        
        # Strategy: Mix samples where Age >= 70 OR SpO2 is [92-94]
        N1 = count // 2 # 50% Age-based
        N2 = count - N1 # 50% SpO2-based
        
        # Group 1: Age-based Medium
        age_1 = np.random.randint(70, 90, N1) 
        spO2_1 = np.random.randint(95, 100, N1) # Safe SpO2 (avoids High/Medium SpO2 triggers)

        # Group 2: SpO2-based Medium
        age_2 = np.random.randint(18, 69, N2) # Safe Age
        spO2_2 = np.random.randint(92, 95, N2) # SpO2 in [92, 94]
        
        # Combine parameters and ensure other High triggers are avoided
        age = np.concatenate([age_1, age_2])
        spO2 = np.concatenate([spO2_1, spO2_2])
        
        heart_rate = np.random.randint(50, 131, count) # HR <= 130
        bp_systolic = np.random.randint(95, 180, count) # SBP >= 95
        bp_diastolic = np.random.randint(60, 110, count)
        temperature = np.round(np.random.uniform(36.0, 39.9, count), 1) # Temp < 40.0
        
    elif severity_label == 'high':
        # Must meet at least ONE High criteria.
        
        # Strategy: Ensure one of the main high-risk indicators is met for different subsets
        N1 = count // 3 # SpO2-based High (< 92)
        N2 = count // 3 # HR-based High (> 130)
        N3 = count - N1 - N2 # Temp-based High (>= 40.0)
        
        # SpO2-based High
        spO2_1 = np.random.randint(85, 92, N1)
        hr_1 = np.random.randint(50, 130, N1)
        temp_1 = np.round(np.random.uniform(36.0, 39.9, N1), 1)
        
        # HR-based High
        spO2_2 = np.random.randint(95, 100, N2)
        hr_2 = np.random.randint(131, 160, N2)
        temp_2 = np.round(np.random.uniform(36.0, 39.9, N2), 1)

        # Temp-based High
        spO2_3 = np.random.randint(95, 100, N3)
        hr_3 = np.random.randint(50, 130, N3)
        temp_3 = np.round(np.random.uniform(40.0, 40.5, N3), 1)

        # Combine parameters
        spO2 = np.concatenate([spO2_1, spO2_2, spO2_3])
        heart_rate = np.concatenate([hr_1, hr_2, hr_3])
        temperature = np.concatenate([temp_1, temp_2, temp_3])
        
        age = np.random.randint(18, 90, count) 
        bp_systolic = np.random.randint(90, 180, count)
        bp_diastolic = np.random.randint(60, 110, count)

    # Create DataFrame for the generated batch
    df_class = pd.DataFrame({
        'age': age,
        'heart_rate': heart_rate,
        'bp_systolic': bp_systolic,
        'bp_diastolic': bp_diastolic,
        'spO2': spO2,
        'temperature': temperature,
        'severity': severity_label
    })
    
    return df_class

# --- 4. Generate and Combine the new data ---
df_low = generate_class_data('low', TARGET_COUNTS['low'])
df_medium = generate_class_data('medium', TARGET_COUNTS['medium'])
df_high = generate_class_data('high', TARGET_COUNTS['high'])

df_new_synthetic = pd.concat([df_low, df_medium, df_high], ignore_index=True)

# Shuffle the final dataset
df_new_synthetic = df_new_synthetic.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the new dataset
output_filename = 'new_random_synthetic_data.csv'
df_new_synthetic.to_csv(output_filename, index=False)