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
df_train, df_val, df_test, df_geracao, scaler = data_pre_processing()
print(df_geracao)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
X_val, y_val = df_to_X_y(len_seq[3], df_val)
X_test, y_test = df_to_X_y(len_seq[3], df_test)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#Applying the model
model = TransformerModel(input_size=1, seq_len=len_seq[3]).to(device)

#Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 10 # Número de épocas sem melhoria para parar (early stopping)

print("\nIniciando Treinamento...")
for epoch in range(100):
    model.train()
    batch_train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Move data para o device

        optimizer.zero_grad()
        outputs = model(X_batch) # Shape: (batch, n_steps_out)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())

    epoch_train_loss = np.mean(batch_train_losses)
    train_losses.append(epoch_train_loss)

    # Validação
    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            batch_val_losses.append(loss.item())

    epoch_val_loss = np.mean(batch_val_losses)
    val_losses.append(epoch_val_loss)

    print(f'Epoch [{epoch+1}/{100}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')

    # Early Stopping e Salvamento do Melhor Modelo
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_transformer_model.pth')
        print(f'  -> Melhor modelo salvo com Val Loss: {best_val_loss:.6f}')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'  -> Early stopping triggered após {epoch+1} épocas.')
            break

# Carregar o melhor modelo salvo
print("\nCarregando o melhor modelo treinado...")
model.load_state_dict(torch.load('best_transformer_model.pth'))

#======================================================================================#
#                                   --Evaluation--                                     #
mse = MeanSquaredError()
r2_metric = R2Score()
model.eval()
predictions = []
with torch.no_grad():
  for batch in test_loader:
    X_batch, y_batch = batch
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
        X_batch = X_batch.to(device)
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

#======================================================================================#

#predicting future values
def enable_dropout(model):
    """ Enable dropout layers during inference """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

T = 50  # Monte Carlo samples
H = 10  # Horizon (future steps)
model.eval()
enable_dropout(model)  # Dropout active at inference

future_means = []
future_stds = []

last_sequence_data = df_geracao.tail(len_seq[3]).to_numpy()
last_sequence = torch.tensor(last_sequence_data, dtype=torch.float32).view(1, len_seq[3], 1).to(device)

with torch.no_grad():
    for _ in range(H):
        predictions_t = []
        for _ in range(T):
            output = model(last_sequence)  # shape: (1, 1)
            predictions_t.append(output.cpu().numpy())

        predictions_t = np.array(predictions_t).squeeze()  # shape: (T,)
        mean_pred = predictions_t.mean()
        std_pred = predictions_t.std()

        # Store
        future_means.append(mean_pred)
        future_stds.append(std_pred)

        # Use mean prediction to roll forward
        next_input = torch.tensor([[mean_pred]], dtype=torch.float32).to(device)
        last_sequence = torch.cat((last_sequence[:, 1:, :], next_input.unsqueeze(1)), dim=1)

# Denormalize
future_means = np.array(future_means).reshape(-1, 1)
future_stds = np.array(future_stds).reshape(-1, 1)

future_means_denorm = scaler.inverse_transform(future_means)
future_stds_denorm = scaler.scale_ * future_stds  # std scales linearly with original scale

print("Future 10 predictions (denormalized, with uncertainty):")
for i in range(H):
    print(f"Step {i+1}: {future_means_denorm[i][0]:.2f} ± {2*future_stds_denorm[i][0]:.2f}")

# Combine and plot
combined_data = np.concatenate([actual_denormalized[-100:], future_means_denorm], axis=0)
df = pd.DataFrame(combined_data, columns=['Geração de Energia'])

dfdata = df[:100]
dfpred = df[99:]
lower = future_means_denorm.flatten() - 2 * future_stds_denorm.flatten()
upper = future_means_denorm.flatten() + 2 * future_stds_denorm.flatten()

# Plot with uncertainty
fig4 = plt.figure(figsize=(15, 7))
plt.plot(dfdata.index, dfdata['Geração de Energia'], color='blue', label='Valores Reais')
plt.plot(dfpred.index, dfpred['Geração de Energia'], color='red', label='Valores Previstos (média)')
plt.fill_between(dfpred.index[1:], lower, upper, color='gray', alpha=0.4, label='±2σ Confiança')
plt.title('Continuation of Test Data with MC Dropout Future Predictions')
plt.xlabel('Time Step')
plt.ylabel('Energy Generation')
plt.legend()
plt.grid(True)
plt.show()
fig4.savefig('10 Steps Prediction',dpi = 300)