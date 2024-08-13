import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

arima_order_dict = {
    'ma': (0, 0, 5),
    'ar': (13, 0, 0),
    'arma': (13, 0, 5),
    'arima': (13, 1, 5),
}

def train_arima_model_family(data,model_name):
    print(f" for {model_name} selected params: {arima_order_dict[model_name]}")
    model = ARIMA(data['Close'], order=arima_order_dict[model_name])
    print("model training...")
    model = model.fit()
    print("model training completed.")
    return model


