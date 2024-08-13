import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.data.data_loader import create_data_loader
from src.models.nn_models import FeedForwardModel, CNNModel, TransformerDecoderModel, RNNModel, LSTMModel
from src.models.ml_models import train_ml_models
from src.models.stats_models import train_arima_model_family

seq_length = 50
batch_size = 8
hidden_size = 30
num_layers = 2
nhead = 2
num_epochs = 100
learning_rate = 0.001

deep_learning_models = {
    'rnn': RNNModel,
    'ffn': FeedForwardModel,
    'cnn': CNNModel,
    'lstm': LSTMModel,
}

ml_models = ['linear_regression', 'svr', 'knn', 'random_forest']
statistical_models = ['ma', 'ar', 'arma', 'arima']


def train_model(data, model_type):
    if model_type in deep_learning_models:
        x,y = data
        input_size = x.shape[-1]
        data_loader = create_data_loader(x,y, batch_size)
        model = deep_learning_models[model_type](input_size,hidden_size)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for inputs, targets in data_loader:
                if model_type == 'cnn':
                    inputs = inputs.unsqueeze(1)  # CNN expects 3D input: (batch_size, channels, seq_length)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        return model

    elif model_type in ml_models:
        X_train, y_train = data
        model = train_ml_models(model_type,X_train, y_train)
        return model

    elif model_type in statistical_models:
        print(f"Going to instantiate stats model: {model_type}")
        model = train_arima_model_family(data,model_type)
        return model

    else:
        raise ValueError("Invalid model_type.")

