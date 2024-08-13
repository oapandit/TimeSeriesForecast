import torch
import pandas as pd
import numpy as np
import os
from src.data.data_loader import create_data_loader

pred_data_root = "/Users/onkar.pandit/PycharmProjects/dummy/TimeSeriesForecast/data/predictions"

def rmse(df,pred_col_name):
    try:
        mse = round(np.sqrt(np.sum((df['Close'] - df[pred_col_name])**2)/len(df)),2)
        return mse
    except:
        print(f"Error while calculating rmse for: {pred_col_name}")
        return -1

def evaluate_model(model, data, model_type):
    if model_type in ['rnn','ffn', 'cnn', 'lstm']:
        model.eval()
        x, y,df_test = data
        predictions = []
        data_loader = create_data_loader(x, y, batch_size=8)

        with torch.no_grad():
            for inputs, targets in data_loader:
                if model_type == 'cnn':
                    inputs = inputs.unsqueeze(1)  # CNN expects 3D input: (batch_size, channels, seq_length)
                outputs = model(inputs)
                predictions.extend([o[0] for o in outputs.tolist()])
        pred_col_name = f'{model_type}_Close_pred'
        df_test[pred_col_name] = predictions
        return df_test

    elif model_type in ['linear_regression', 'svr', 'knn', 'random_forest']:
        X_test, y_test,df_test = data
        predictions = model.predict(X_test)
        true_values = y_test
        pred_col_name = f'{model_type}_Close_pred'
        df_test[pred_col_name] = predictions
        return df_test

    elif model_type in ['ar', 'ma', 'arma', 'arima']:
        pred_col_name = f'{model_type}_Close_pred'
        forecast = model.forecast(steps=len(data))
        data[pred_col_name] = forecast
        return data

    else:
        raise NotImplementedError



def evaluate_and_save_results(model, data, model_type,stock_name):
    data = evaluate_model(model, data, model_type)
    data.to_csv(os.path.join(pred_data_root,f'{stock_name}_{model_type}.csv'), index=False)
