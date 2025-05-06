from helpers import *
from imblearn.pipeline import Pipeline as imb_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from typing import Callable

## Optuna objective function
def objective(trial: Trial, x: np.ndarray | pd.DataFrame, y: np.ndarray, pipeline: Callable, cv: int | StratifiedKFold = 5):
    """
    Objective function for Optuna to optimize

    Args:
        trial (Trial): Optuna trial
        x (np.ndarray | pd.DataFrame): dependent variable
        y (np.ndarray): independent variable
        pipeline (Callable): pipeline wrapper with trial as parameter
        cv (int | StratifiedKFold, optional): cv splits. Defaults to 5.

    Returns:
        _type_: pr_auc score of pipeline (model)
    """
    model = pipeline(trial)

    score = cross_val_score(model, x, y, n_jobs=-1, cv=cv,
                            scoring="average_precision")
    pr_auc = score.mean()

    return pr_auc

## loan_acceptance pipelines
def acc_xgb_smote_pipeline(trial: Trial) -> Pipeline:
    """
    xgboost + smote pipeline for loan acceptance prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial)), ("scaler", scalers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, acc_num_cols),
            ("categorcal_pipeline", TargetEncoder(), acc_cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = imb_pipeline(
        [
            ("column_transformer", column_transformer),
            ("smote", smote(trial)),
            ("model", xgb_clf(trial)),
        ]
    )

    return pipeline

def acc_balanced_rf_pipeline(trial):
    """
    BalancedRandomForest pipeline for loan acceptance prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial)), ("scaler", scalers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, acc_num_cols),
            ("categorcal_pipeline", TargetEncoder(), acc_cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", balanced_rf(trial)),
        ]
    )

    return pipeline

def acc_xgb_balanced_pipeline(trial):
    """
    xgboost class_balanced pipeline for loan acceptance prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial)), ("scaler", scalers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, acc_num_cols),
            ("categorcal_pipeline", TargetEncoder(), acc_cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", balanced_xgb_clf(trial)),
        ]
    )

    return pipeline

def acc_xgb_balanced_imp_free_pipeline(trial):
    """
    xgboost class_balanced with internal imputation pipeline for loan acceptance prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(steps=[("scaler", scalers(trial))])

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, acc_num_cols),
            ("categorcal_pipeline", TargetEncoder(), acc_cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", balanced_xgb_clf(trial)),
        ]
    )

    return pipeline


## grade prediction pipelines

def grade_xgb_smote_pipeline(trial):
    """
    xgboost + smote pipeline for grade prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial)), ("scaler", scalers(trial))]
    )

    init_ls_status_process = trial.suggest_categorical(
        "encode", ["ohe", "target"])

    column_transformer = ""

    if init_ls_status_process == "ohe":
        column_transformer = ColumnTransformer(
            transformers=[
                ("numerical_pipeline", num_pipe, grade_num_cols),
                ("feat_purpose_trans", TargetEncoder(), ["purpose"]),
                (
                    "feat_initial_list_status_trans",
                    OneHotEncoder(),
                    ["initial_list_status"],
                ),
            ],
            remainder="passthrough",
        )

    else:
        column_transformer = ColumnTransformer(
            transformers=[
                ("numerical_pipeline", num_pipe, grade_num_cols),
                ("categorical_pipeline", TargetEncoder(), grade_cat_cols),
            ],
            remainder="passthrough",
        )

    pipeline = imb_pipeline(
        [
            ("column_transformer", column_transformer),
            ("smote", smote(trial)),
            ("model", xgb_clf(trial)),
        ]
    )

    return pipeline


def grade_balanced_rf_pipeline(trial):
    """
    BalancedRandomForest pipeline for grade prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial)), ("scaler", scalers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, grade_num_cols),
            ("categorcal_pipeline", TargetEncoder(), grade_cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", balanced_rf(trial)),
        ]
    )

    return pipeline

sub_grade_num_cols = grade_num_cols + ['grade']

def sub_grade_balanced_rf_pipeline(trial):
    """
    BalancedRandomForest pipeline for sub_grade prediction

    Args:
        trial (Trial): Optuna trial

    Returns:
        Pipeline: pipeline
    """

    num_pipe = Pipeline(
        steps=[("imputer", imputers(trial)), ("scaler", scalers(trial))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", num_pipe, sub_grade_num_cols),
            ("categorcal_pipeline", TargetEncoder(), grade_cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("model", balanced_rf(trial)),
        ]
    )

    return pipeline