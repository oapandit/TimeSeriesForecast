from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# Dictionary mapping model types to their training functions
train_model_dict = {
    'linear_regression':  LinearRegression(),
    'svr':  SVR(kernel='rbf'),
    'knn':  KNeighborsRegressor(n_neighbors=5),
    'random_forest':  RandomForestRegressor(n_estimators=10, random_state=42),
}

def train_ml_models(model_type,X_train, y_train,):
    model = train_model_dict[model_type]
    model.fit(X_train, y_train)
    return model


