# ------------------------------------------------------------NOTES--------------------------------------------------------------------------
# using mlflow: https://mlflow.org/docs/latest/ml/tracking/quickstart/
# ------------------------------------------------------------------------------------------------------------------------------------------

import mlflow
from mlflow.models import infer_signature


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
acc = accuracy_score(y_test, y_pred)


mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

mlflow.set_experiment("MLFlow Tutorial")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    signature = infer_signature(X_train, lr.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="mlflow-tutorial"
    )

    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training info":"Basic lr model for sklearn iris data"}
    )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)

    iris_feature_names = datasets.load_iris().feature_names

    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    print(result[:4])
