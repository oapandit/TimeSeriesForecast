import os
import matplotlib.pyplot as plt
import numpy as np
from src.data.utils import load_data, load_pred_data_cumulative
from src.evaluation.evaluate import pred_data_root, rmse

series_names = ['ITC','Infosys','NIFTY50','SENSEX']
plots_dir = os.path.join(pred_data_root,"plots")
def read_stocks_pred_csvs(models):
    itc_df = load_pred_data_cumulative(pred_data_root,series_names[0],models)
    infy_df = load_pred_data_cumulative(pred_data_root, series_names[1], models)
    nifty_df = load_pred_data_cumulative(pred_data_root, series_names[2], models)
    sensex_df = load_pred_data_cumulative(pred_data_root, series_names[3], models)
    return itc_df,infy_df,nifty_df,sensex_df

def plot_predictions(true_values, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.show()


def generate_graph(fig_ax, df, models):
    _df = df[df['year'] >= 2021]
    if np.any(_df['year'] > 2021):
        _df = _df[_df['year'] < 2022]
        _df = _df[_df['month'] < 7]

    x_axis = list(range(len(_df['Date'])))
    # x_axis = df['Date']

    fig_ax.plot(x_axis, _df['Close'], label='Actual')
    for ind, model_name in enumerate(models):
        fig_ax.plot(x_axis, _df[f'{model_name}_Close_pred'], label=model_name.upper())

def generate_prediction_plots(models, model_type):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    axes = axes.flatten()

    itc_df,infy_df,nifty_df,sensex_df = read_stocks_pred_csvs(models)

    for i, ax in enumerate(axes):
        if i == 0:
            generate_graph(ax, itc_df, models)
        elif i == 1:
            generate_graph(ax, infy_df, models)
        elif i == 2:
            generate_graph(ax, nifty_df, models)
        elif i == 3:
            generate_graph(ax, sensex_df, models)

        # Adding labels and title to each subplot
        ax.set_xlabel('Days')
        ax.set_ylabel('Share Price')
        ax.set_title(f'{series_names[i]}')
        ax.legend()

    fig.suptitle('Actual vs Predicted')
    fig.tight_layout()

    plt.savefig(os.path.join(plots_dir,f'{model_type}_act_pred.png'))
    # plt.show()

def get_mse_for_model_types(models,df):
    mse_for_stock = [rmse(df,f'{model_name}_Close_pred') for model_name in models]
    return mse_for_stock

def generate_error_plots(models,model_type):
    itc_df, infy_df, nifty_df, sensex_df = read_stocks_pred_csvs(models)
    mse_list = [get_mse_for_model_types(models,df) for df in [itc_df, infy_df, nifty_df, sensex_df]]

    # Setting the bar width
    bar_width = 0.85

    c = ['orange', 'green', 'red', 'purple']

    bar_positions = [m.upper() for m in models]

    # Creating a 2x2 subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # Plotting the first bar chart in the top-left subplot
    axes[0, 0].bar(bar_positions, mse_list[0], width=bar_width, label='Group 1', color=c)
    axes[0, 0].set_title(series_names[0])
    axes[0, 0].set_ylabel('Error')

    for i, value in enumerate(mse_list[0]):
        axes[0, 0].text(i, value + 0.5, str(value), ha='center', va='bottom')

    # Plotting the second bar chart in the top-right subplot
    axes[0, 1].bar(bar_positions, mse_list[1], width=bar_width, label='Group 2', color=c)
    axes[0, 1].set_title(series_names[1])
    axes[0, 1].set_ylabel('Error')

    for i, value in enumerate(mse_list[1]):
        axes[0, 1].text(i, value + 0.5, str(value), ha='center', va='bottom')

    # Plotting the third bar chart in the bottom-left subplot
    axes[1, 0].bar(bar_positions, mse_list[2], width=bar_width, label='Group 3', color=c)
    axes[1, 0].set_title(series_names[2])
    axes[1, 0].set_ylabel('Error')

    for i, value in enumerate(mse_list[2]):
        axes[1, 0].text(i, value + 0.5, str(value), ha='center', va='bottom')

    # Plotting the fourth bar chart in the bottom-right subplot
    axes[1, 1].bar(bar_positions, mse_list[3], width=bar_width, label='Group 4', color=c)
    axes[1, 1].set_title(series_names[3])
    axes[1, 1].set_ylabel('Error')

    for i, value in enumerate(mse_list[3]):
        axes[1, 1].text(i, value + 0.5, str(value), ha='center', va='bottom')

    # Adding labels and title for the entire figure
    fig.suptitle('Prediction Errors')

    # Adjust layout to prevent clipping of titles
    fig.tight_layout()

    plt.savefig(os.path.join(plots_dir,f'{model_type}_pred_err.png'))
    # Display the plot
    # plt.show()

def generate_error_comparison_between_model_families(model_families_dict):
    model_family_to_best_error_model_dict = {}

    best_error = None
    best_model = None

    for model_type,models in model_families_dict.items():
        itc_df, infy_df, nifty_df, sensex_df = read_stocks_pred_csvs(models)

        mse_list = [get_mse_for_model_types(models, df) for df in [itc_df, infy_df, nifty_df, sensex_df]]

        best_error_list_in_current_model_family_for_four_stocks = np.min(mse_list,axis=1)
        best_model_in_current_model_family_for_four_stocks = np.argmin(mse_list,axis=1)

        model_family_to_best_error_model_dict[model_type] = [best_error_list_in_current_model_family_for_four_stocks,
                                                             best_model_in_current_model_family_for_four_stocks]

        if best_error is None:
            best_error = best_error_list_in_current_model_family_for_four_stocks
            best_model = best_model_in_current_model_family_for_four_stocks
        else:
            best_error = np.vstack([best_error,best_error_list_in_current_model_family_for_four_stocks])
            best_model = np.vstack([best_model,best_model_in_current_model_family_for_four_stocks])

    best_error_per_stock = best_error.transpose()
    best_model_per_stock = best_model.transpose()

    # Setting the bar width
    bar_width = 0.85

    c = ['orange', 'green', 'red']

    # Creating a 2x2 subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axes = axes.flatten()

    for i,ax in enumerate(axes):

        bar_positions = []

        bar_positions.append(model_families_dict['statistical_models'][best_model_per_stock[i][0]])
        bar_positions.append(model_families_dict['ml_models'][best_model_per_stock[i][1]])
        bar_positions.append(model_families_dict['deep_learning_models'][best_model_per_stock[i][2]])

        ax.bar(bar_positions, best_error_per_stock[i], width=bar_width, label='Group 1', color=c)
        ax.set_title(series_names[i])
        ax.set_ylabel('Error')

        for i, value in enumerate(best_error_per_stock[i]):
            ax.text(i, value + 0.5, str(value), ha='center', va='bottom')

    fig.suptitle('Best Models')

    # Adjust layout to prevent clipping of titles
    fig.tight_layout()

    plt.savefig(os.path.join(plots_dir, f'all_model_comparisons.png'))
    # Display the plot
    # plt.show()





