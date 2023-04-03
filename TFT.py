import torch
import numpy as np
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

# Load data
data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

# Define input and target variables
max_encoder_length = 24  # length of the encoder sequence
max_prediction_length = 6  # length of the prediction horizon
training_cutoff = 2000  # index to split data into train/validation/test sets

# Define dataset
dataset = TimeSeriesDataSet(
    data[:training_cutoff],
    time_idx=0,  # index of the time dimension in the data array
    target="Y",  # name of the target variable
    group_ids=["id"],  # name of the variable indicating the time series group
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["A", "B"],
    static_reals=["C"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["A", "B"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["Y"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Define dataloaders
batch_size = 128
train_dataloader = dataset.to_dataloader(
    train=True, batch_size=batch_size, num_workers=2)
val_dataloader = dataset.to_dataloader(
    train=False, batch_size=batch_size, num_workers=2)

# Define model
tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,
    loss=torch.nn.L1Loss(),
    log_interval=10,
)

# Train model
trainer = pl.Trainer(gpus=1, gradient_clip_val=0.1)
trainer.fit(tft, train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader)

# Make predictions on test data
test_data = data[training_cutoff:]
test_dataset = TimeSeriesDataSet.from_dataset(dataset, test_data, predict=True)
test_dataloader = test_dataset.to_dataloader(
    batch_size=batch_size, num_workers=2)
predictions = tft.predict(test_dataloader)

# Evaluate model performance
mse_loss = torch.nn.MSELoss()(torch.tensor(predictions),
                              torch.tensor(test_dataset["Y"]))
mae_loss = torch.nn.L1Loss()(torch.tensor(predictions),
                             torch.tensor(test_dataset["Y"]))
print(f"MSE loss: {mse_loss:.2f}, MAE loss: {mae_loss:.2f}")
