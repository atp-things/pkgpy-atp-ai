import argparse
import logging
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()


# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # parameters
    alpha = args.alpha
    l1_ratio = args.l1_ratio
    # np.random.seed(40)

    # setup mlflow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    print("The set tracking uri is ", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experment_1")

    # Read the wine-quality csv file from local
    data = pd.read_csv("data/winequality.csv", header=0, sep=";")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    with mlflow.start_run(experiment_id=exp.experiment_id):
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            # random_state=42,
        )
        model.fit(train_x, train_y)

        predicted_qualities = model.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")
        metrics = {"rmse": rmse, "r2": r2, "mae": mae}
        params = {"alpha": alpha, "l1_ratio": l1_ratio}
        # mlflow.log_param("alpha", alpha)
        # mlflow.log_param("l1_ratio", l1_ratio)
        # mlflow.log_metric("rmse", rmse)
        # mlflow.log_metric("r2", r2)
        # mlflow.log_metric("mae", mae)
        mlflow.set_tags({"version": "v1", "priority": "p1"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("data/artifact.json", "artifacts")

        mlflow.sklearn.log_model(model, "model")
