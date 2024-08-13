import pandas as pd
import os

data_root = "/Users/onkar.pandit/PycharmProjects/dummy/TimeSeriesForecast/data/raw/"

stock_to_csv = {
    'Infosys': os.path.join(data_root,"INFY.csv"),
    'ITC': os.path.join(data_root,"ITC.csv"),
    'NIFTY50': os.path.join(data_root,'NIFTY_50.csv'),
    'SENSEX': os.path.join(data_root,'sensex.csv')
}
def load_data(file_path):
    return pd.read_csv(file_path, index_col=False)

def load_pred_data_cumulative(pred_data_root,stock_name,model_names):
    df_combined = None
    for m in model_names:
        file_path = os.path.join(pred_data_root,f"{stock_name}_{m}.csv")
        df = load_data(file_path)
        if df_combined is None:
            df_combined = df
        else:
            df_combined[f'{m}_Close_pred'] = df[f'{m}_Close_pred']
    return df_combined

def extract_year_month(df):
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['month'] = pd.DatetimeIndex(df['Date']).month
    return df

def download_data():
    # Code to download data from a source
    pass

