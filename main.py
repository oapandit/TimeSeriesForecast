from src.data.data_loader import stock_data_loader

from src.models.train import train_model
from src.evaluation.evaluate import evaluate_and_save_results
from src.visualization.visualize import generate_prediction_plots, generate_error_plots, generate_error_comparison_between_model_families

series_names = ['ITC', 'Infosys', 'NIFTY50', 'SENSEX']
# series_names = ['ITC']


statistical_models = ['ma', 'ar', 'arma', 'arima']
ml_models = ['linear_regression', 'svr', 'knn', 'random_forest']
deep_learning_models = ['rnn', 'ffn', 'cnn', 'lstm']
# deep_learning_models = ['rnn']

# model_types = statistical_models
# model_types = ml_models
model_types = deep_learning_models

# model_types = statistical_models + ml_models + deep_learning_models

model_families_dict = {
'statistical_models': ['ma', 'ar', 'arma', 'arima'],
'ml_models':['linear_regression', 'svr', 'knn', 'random_forest'],
'deep_learning_models': ['rnn', 'ffn', 'cnn', 'lstm'],
}

def main():
    for stock_name in series_names:
        print(f"\n\n\n\n*** STOCK: {stock_name} ****\n\n")
        data = None
        for model_type in model_types:
            print(f"\n*** Training {model_type} ****\n")
            if data is None:
                data_train, data_test = stock_data_loader(stock_name, model_type)
            model = train_model(data_train, model_type)
            evaluate_and_save_results(model, data_test, model_type, stock_name)

    # generate_prediction_plots(statistical_models,model_type='stats_models')
    # generate_prediction_plots(ml_models,model_type='ml_models')
    generate_prediction_plots(deep_learning_models, model_type='deep_learning_models')

    generate_error_plots(deep_learning_models, model_type ='deep_learning_models')
    # generate_error_plots(ml_models, model_type='ml_models')
    # generate_error_plots(statistical_models,model_type='stats_models')

    generate_error_comparison_between_model_families(model_families_dict)


if __name__ == "__main__":
    main()

    # model_type = 'arima'  # Change to 'feedforward', 'cnn', 'transformer', 'linear_regression', 'svr', 'knn', 'random_forest', 'ar', 'ma', 'arma' as needed
    #
    # model = train_model(data['Close'].values, model_type)
    #
    # # Evaluation
    # if model_type in ['ar', 'ma', 'arma', 'arima']:
    #     data_loader = data['Close'].values[-seq_length:]
    # else:
    #     data_loader = (X_test, y_test) if model_type in ['linear_regression', 'svr', 'knn', 'random_forest'] else create_data_loader(data['Close'].values, seq_length, batch_size)
    # true_values, predictions = evaluate_model(model, data_loader, model_type)
