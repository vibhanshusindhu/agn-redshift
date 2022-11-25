import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import autosklearn.regression
import autosklearn.metrics


def get_data(filename: str) -> tuple:
    data = pd.read_csv(filename, index_col=0)
    data.drop(columns="X.name", inplace=True)
    y = data.pop("z")
    X = data
    return X, y


def plot_predictions(train_predictions: np.array, test_predictions: np.array):
    plt.scatter(train_predictions, y_train, label="Train samples", c="#d95f02")
    plt.scatter(test_predictions, y_test, label="Test samples", c="#7570b3")
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.legend()
    plt.plot([0, 4], [0, 4], c="k", zorder=0)
    plt.xlim([0, 4])
    plt.ylim([0, 4])
    plt.tight_layout()
    plt.savefig("true_vs_predicted_rmse.png")


if __name__ == "__main__":
    X, y = get_data("FullData.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=3600,
        per_run_time_limit=60,
        ensemble_size=30,
        ensemble_nbest=6,
        metric=autosklearn.metrics.root_mean_squared_error,
        disable_evaluator_output=False,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": 10},
        tmp_folder="/tmp/autosklearn_regression_resampling_cv10_rmse_parallel_agn",
        n_jobs=6,
        memory_limit=4084,
        seed=1,
    )
    automl.fit(X_train, y_train, dataset_name="agn")

    print(automl.leaderboard())
    print(automl.sprint_statistics())
    results = pd.DataFrame(automl.cv_results_)
    results.to_csv("automl_rmse_cv10_fulldata09.csv")

    # refit
    automl.refit(X_train.copy(), y_train.copy())
    train_predictions = automl.predict(X_train)
    test_predictions = automl.predict(X_test)

    print("Train R2 score:", r2_score(y_train, train_predictions))
    print("Test R2 score:", r2_score(y_test, test_predictions))
    print("Train MAE score:", mean_absolute_error(y_train, train_predictions))
    print("Test MAE score:", mean_absolute_error(y_test, test_predictions))
    print("Train RMSE score:", np.sqrt(mean_squared_error(y_train, train_predictions)))
    print("Test RMSE score:", np.sqrt(mean_squared_error(y_test, test_predictions)))
