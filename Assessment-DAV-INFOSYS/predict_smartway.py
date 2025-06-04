import pandas as pd
import joblib

# Load the model
model = joblib.load("smartway_decision_tree.pkl")

# Create a DataFrame with 7 features (match your training data)
new_vehicle = pd.DataFrame([{
    'Fuel': 'Gasoline',
    'Engine_Cylinders': 4,
    'Greenhouse_Gas_Score': 3,
    'Air_Pollution_Score': 4,
    'Combined_Mpg': 10,
    'Engine_Displacement_L': 3.7,
    'Drive': 4
}])

# Predict
prediction = model.predict(new_vehicle)

label_map = {0: "No", 1: "Yes", 2: "Elite"}
print("Prediction:", label_map[prediction[0]])
