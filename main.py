import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torchmetrics import MeanSquaredError, R2Score
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datapreprocessing import data_pre_processing
from tst import PositionalEncoding,TransformerModel


#Variable Values
df_train, df_test, df_geracao, scaler = data_pre_processing()

#ACF and PACF
fig2 = plot_acf(df_geracao, lags=20)
plt.title('Autocorrelation Function (ACF)')
plt.show()

fig1 = plot_pacf(df_geracao, lags=20)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# fig1.savefig("ACF.png", dpi=300)
# fig2.savefig("PACF.png", dpi=300)

#Significant lags
len_seq = [3,6,8,15] #due to PACF

#creating training and testing dataset with torch and selecting the lags 
def df_to_X_y(len_seq, obs):
  df_as_np = np.array(obs)
  X = []
  y = []
  for i in range(len(obs)-len_seq):
    row = [[a] for a in df_as_np[i:i+len_seq]]
    X.append(row)
    label = df_as_np[i+len_seq]
    y.append(label)
  return torch.tensor(X, dtype=torch.float32).view(-1, len_seq, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

X_train, y_train = df_to_X_y(len_seq[3], df_train)
X_test, y_test = df_to_X_y(len_seq[3], df_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#Applying the model
model = TransformerModel(input_size=1, seq_len=len_seq[3]).to('cuda')

#Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# sched_get_priority_max

#Defining number ef epochs
num_epochs = 100

#Training loop
for epoch in range(num_epochs):
  model.train()
  for batch in train_loader:
    X_batch, y_batch = batch
    X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
  print(f'Epoch: {epoch} Loss: {loss}')

#Evaluation
mse = MeanSquaredError()
r2_metric = R2Score()
model.eval()
predictions = []
with torch.no_grad():
  for batch in test_loader:
    X_batch, y_batch = batch
    X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
    outputs = model(X_batch)
    predictions.append(outputs.cpu().numpy())
    mse = criterion(outputs, y_batch)
    r2_value = r2_metric(outputs, y_batch)
    r2_value = r2_value.cpu()

prediction = np.concatenate(predictions, axis=0)
print(f"MSE: {mse.item():.4f}, RMSE: {mse.sqrt():.4f}, R²: {r2_value:.4f}")


model.eval()
predictions_normalized = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to('cuda')
        outputs = model(X_batch)
        predictions_normalized.append(outputs.cpu().numpy())

predictions_normalized = np.concatenate(predictions_normalized, axis=0)
predictions_denormalized = scaler.inverse_transform(predictions_normalized)
actual_denormalized = scaler.inverse_transform(y_test.cpu().numpy())

fig3 = plt.figure(figsize=(12, 6))
plt.plot(actual_denormalized, label='Actual')
plt.plot(predictions_denormalized, label='Predicted')
plt.title('Actual vs Predicted Energy Generation (Denormalized)')
plt.xlabel('Time')
plt.ylabel('Energy Generation')
plt.legend()
plt.show()
fig3.savefig("testpredictions.png", dpi=300)