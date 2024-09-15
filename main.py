import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from model import LinearRegression
import plot_utils

# Load the data
train_df = pd.read_csv('train.csv')

# Split the train data into features (X) and target (y)
X = train_df.drop(columns=['ID', 'medv'])
y = train_df['medv']

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
test=pd.read_csv('test.csv')
X_test = test.drop(columns=['ID'])

print(X_test.shape)
# Initialize and train the model
model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)


# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Standardize the test dataset using the same scaler fitted on training data
X_test_scaled = scaler.transform(X_test)
y_test_pred= model.predict(X_test)

# Calculate RMSE for the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE on validation set: {rmse_val}")

# Plotting loss over iterations
plot_utils.plot_loss(model.loss_history)

# Visualize predicted vs actual values for training and validation sets
plot_utils.plot_actual_vs_predicted(y_train, y_train_pred, "Training Data: Actual vs Predicted")
plot_utils.plot_actual_vs_predicted(y_val, y_val_pred, "Validation Data: Actual vs Predicted")

# Predict on the test set
y_test_pred = model.predict(X_test_scaled)

# Create a DataFrame with the test predictions
test_predictions_df = pd.DataFrame({
    'ID': test['ID'],   # Assuming 'ID' column exists in test.csv
    'Predicted_medv': y_test_pred
})

# Save the predictions to a CSV file
test_predictions_df.to_csv('test_predictions.csv', index=False)
print("Test predictions saved to 'test_predictions.csv'.")