# ------------------------------------------------------------NOTES--------------------------------------------------------------------------
# using mlflow: https://mlflow.org/docs/latest/ml/tracking/quickstart/
# ------------------------------------------------------------------------------------------------------------------------------------------

import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

