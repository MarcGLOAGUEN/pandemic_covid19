import pandas as pd
from prophet import Prophet
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

train_size = 0.8

def read_csv(path: str):
    df = pd.read_csv(path).rename(columns={
        "Unnamed: 0": "Date",
        "index": "Date"
    }).set_index("Date")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    return df
    
def prepare_dataset(df: pd.DataFrame):
    """
    :param pd.DataFrame df: df must have datetime on first column and value on 2nd column
    """
    df.columns = ["ds", "y"]
    size = int(train_size*len(df))
    train = df.iloc[:size]
    test = df.iloc[size:]
    return train, test

def pred_prophet(train: pd.DataFrame, periods: int):
    """
    Instanciate a Prophet model fitted on train and make predictions for *periods*
    """
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast

def plot_test_pred(test, forecast):
    forecast = forecast[["ds", "yhat"]]
    df = test.merge(forecast).set_index("ds")
    fig = px.line(df)
    return fig
    
    