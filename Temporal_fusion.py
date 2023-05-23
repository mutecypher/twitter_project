import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming TS3 is a pandas DataFrame object with 'timestamp' and 'measurement' columns

# Step 1: Split TS3 into training, validation, and test sets
train, test = train_test_split(TS3, test_size=0.2, shuffle=False)
train, val = train_test_split(train, test_size=0.2, shuffle=False)

# Step 2: Define the TimeSeriesDataSet
training_data = TimeSeriesDataSet(
    train,
    target="measurement",
    time_idx="timestamp",
    min_encoder_length=10,  # Minimum history length for each sample
    max_encoder_length=50,  # Maximum history length for each sample
    min_prediction_length=1,  # Minimum forecast length for each sample
    max_prediction_length=10,  # Maximum forecast length for each sample
)

# Step 3: Create the TFT model
tft = TemporalFusionTransformer.from_dataset(training_data)

# Step 4: Train the model
trainer = tft.train_dataloader(training_data, batch_size=64)
tft.fit(trainer, epochs=10)

# Step 5: Predict on the validation set
validation_data = TimeSeriesDataSet.from_dataset(training_data, val, predict=True)
val_dataloader = validation_data.to_dataloader(batch_size=32, shuffle=False)
val_predictions = tft.predict(val_dataloader)

# Calculate root mean squared error (RMSE) on the validation set
val_rmse = mean_squared_error(validation_data.target.numpy(), val_predictions.numpy(), squared=False)
print("Validation RMSE:", val_rmse)

# Step 6: Predict on the test set
test_data = TimeSeriesDataSet.from_dataset(training_data, test, predict=True)
test_dataloader = test_data.to_dataloader(batch_size=32, shuffle=False)
test_predictions = tft.predict(test_dataloader)

# Calculate root mean squared error (RMSE) on the test set
test_rmse = mean_squared_error(test_data.target.numpy(), test_predictions.numpy(), squared=False)
print("Test RMSE:", test_rmse)
