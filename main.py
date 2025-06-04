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
print(df_geracao)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)


num_epochs = 100
early_stop_count = 0
min_val_loss = float('inf')

for epoch in range(num_epochs):
  model.train()
  val_losses = []
  for batch in train_loader:
    X_batch, y_batch = batch
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    val_losses.append(loss.item())
    optimizer.step()
  
  val_loss = np.mean(val_losses)
  scheduler.step(val_loss)
  if val_loss < min_val_loss:
    min_val_loss = val_loss
    early_stop_count = 0
  else:
    early_stop_count += 1
  if early_stop_count >= 5:
    print("Early stopping triggered.")
    break
  print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

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

#fig3.savefig("testpredictions.png", dpi=300)

future_predictions_normalized = []

last_sequence_data = df_geracao.tail(len_seq[3]).to_numpy()
last_sequence = torch.tensor(last_sequence_data, dtype=torch.float32).view(1, len_seq[3], 1).to(device)


with torch.no_grad():
    for _ in range(10):
        next_prediction_normalized = model(last_sequence)
        future_predictions_normalized.append(next_prediction_normalized.cpu().numpy())
        print(f'LS: {last_sequence}')
        print(f'NP: {next_prediction_normalized}')
        last_sequence = torch.cat((last_sequence[:, 1:, :], next_prediction_normalized.unsqueeze(1)), dim=1)

future_predictions_normalized = np.concatenate(future_predictions_normalized, axis=0)
future_predictions_denormalized = scaler.inverse_transform(future_predictions_normalized)

print("Future 10 predictions (denormalized):")
print(future_predictions_denormalized)


combined_data = np.concatenate([actual_denormalized[-100:], future_predictions_denormalized], axis=0)
df = pd.DataFrame(combined_data, columns=['Geração de Energia'])
dfdata = df[:100]
dfpred = df[99:]

fig4 = plt.figure(figsize=(15, 7))
plt.plot(dfdata.index,dfdata['Geração de Energia'], color = 'blue', label = 'Valores Reais')
plt.plot(dfpred.index,dfpred['Geração de Energia'], color = 'red', label = 'Valores Previstos')
plt.title('Continuation of Test Data with Future Predictions')
plt.xlabel('Time Step')
plt.ylabel('Energy Generation')
plt.legend()
plt.grid(True)
plt.show()
# fig4.savefig('10 Steps Prediction',dpi = 300)