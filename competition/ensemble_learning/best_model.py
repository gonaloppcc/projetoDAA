import os
from typing import Dict

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
# We will use the following models for our boosting
from sklearn.ensemble import GradientBoostingClassifier
# We will use the following models for our ensemble learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

TRAINING_DATASET_SOURCE = '../datasets/training_data.csv'
TEST_DATASET_SOURCE = '../datasets/test_data.csv'

os.environ["OMP_NUM_THREADS"] = '5'

categorical_to_numerical = {
    'luminosity': {
        'LOW_LIGHT': 0,
        'LIGHT': 0,
        'DARK': 1
    },
}

"""
'avg_rain': {
    'Sem Chuva': 0,
    'chuva fraca': 1,
    'chuva moderada': 1,
    'chuva forte': 1,
}"""


def convert_record_date(df: pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()

    record_date = pd.DatetimeIndex(df_['record_date'])

    df_.drop('record_date', axis=1, inplace=True)

    df_['hour'] = record_date.hour
    df_['day'] = record_date.day
    df_['month'] = record_date.month
    df_['weekday'] = record_date.weekday
    df_['hour'] = record_date.hour

    return df_


def hour_of_the_day(hour):
    if hour >= 20 or 0 <= hour <= 6:
        return 0
    elif 7 <= hour <= 19:
        return 1

    return 2


# Data Preprocessing
def data_preprocessing(df):
    dropped_columns = ['city_name', 'avg_precipitation', 'magnitude_of_delay', 'avg_rain']

    prep_df = df.drop(dropped_columns, axis=1)
    prep_df.drop_duplicates()
    prep_df.replace(categorical_to_numerical, inplace=True)

    prep_df = convert_record_date(prep_df)

    num_affected_roads = []
    for line in df['affected_roads']:
        unique_roads = set(str(line).split(','))
        valid_roads = [elem for elem in unique_roads if elem != '']
        count = len(valid_roads)
        num_affected_roads.append(count)
    prep_df['num_affected_roads'] = num_affected_roads
    prep_df.drop(columns=['affected_roads'], inplace=True)

    delay_in_minutes = prep_df['delay_in_seconds'].map(lambda seconds: seconds / 60)

    prep_df.drop(columns=['delay_in_seconds'], inplace=True)
    prep_df['delay_in_minutes'] = delay_in_minutes

    est = KBinsDiscretizer(n_bins=2, strategy='kmeans', encode='ordinal')
    prep_df['delay'] = est.fit_transform(prep_df[['delay_in_minutes']])
    prep_df.drop(columns=['delay_in_minutes'], inplace=True)

    prep_df["hour"] = prep_df["hour"].apply(hour_of_the_day)

    numerical_features = [
        'num_affected_roads', 'hour', 'day', 'month', 'weekday', 'avg_temperature', 'avg_atm_pressure',
        'avg_wind_speed', 'avg_humidity'
    ]
    prep_df[numerical_features] = MinMaxScaler().fit_transform(prep_df[numerical_features])

    return prep_df


# Data Splitting
# noinspection PyPep8Naming
def data_splitting(df):
    X = data_preprocessing(df.drop('incidents', axis=1))
    y = df['incidents']

    return train_test_split(X, y, test_size=0.3, random_state=42)


models = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [20],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(gpu_id=0, tree_method='gpu_hist'),
        'params': {
            'n_estimators': [160],
            'max_depth': [5],  # [6, 7, 8],
            'eta': [0.2]  # [0.15]  # [0.1, 0.15, 0.2],
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(),
        'params': {
            'n_estimators': [160],
            'max_depth': [5],
            'learning_rate': [0.2],
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(),
        'params': {
            'n_estimators': [1200, 1500, 2000, 2500],  # [500],  # , 600, 700],
            'max_depth': [3],
            'learning_rate': [0.05],  # , 0.4, 0.5],
            'early_stopping_rounds': [10],
            'verbose': [False]
        }
    },
}

models_actived = {
    'RandomForest': False,
    'XGBoost': False,
    'LightGBM': False,
    'CatBoost': True,
}

models_trained: Dict[str, GridSearchCV] = {}

CV_FOLDS = 4


def print_model_prefix(prefix: str, model_name: str, *text):
    print(f'[{prefix}] <{model_name}>', *text)


def pre_data_preparation(train: pd.DataFrame) -> pd.DataFrame:
    print(train['incidents'].value_counts())

    incidents_count = train['incidents'].value_counts()

    max_count = incidents_count.max()
    print('Max value count:', max_count)

    df_classes = []
    for category, counts in zip(incidents_count.index, incidents_count):
        df_classes.append(train[train['incidents'] == category])

    df_classes_over = []

    for category in df_classes:
        df_classes_over.append(category.sample(max_count, replace=True, random_state=42))

    df_test_over = pd.concat(df_classes_over, axis=0)

    print(df_test_over['incidents'].value_counts())

    return df_test_over


# Model Training
# noinspection PyPep8Naming
def model_training(X, y):
    for model_name, model_obj in models.items():
        model = model_obj['model']
        params = model_obj['params']

        if models_actived[model_name]:
            grid_search = GridSearchCV(estimator=model, param_grid=params, cv=CV_FOLDS, n_jobs=-1,
                                       verbose=1)
            grid_search.fit(X, y)

            print_model_prefix('CV', model_name, "Best Params", grid_search.best_params_)
            print_model_prefix('CV', model_name, "Best Score", grid_search.best_score_)
            print_model_prefix('CV', model_name, "Best Estimator", grid_search.best_estimator_)
            print_model_prefix('Train', model_name, "Accuracy", grid_search.score(X, y))

            models_trained[model_name] = grid_search.best_estimator_


# Model Evaluation
# noinspection PyPep8Naming
def model_evaluation(X_test, y_test) -> (str, GridSearchCV, float):
    best_accuracy = 0
    best_model_evaluation = None
    best_model_name_evaluation = None
    for model_name, model in models_trained.items():
        if models_actived[model_name]:
            y_pred = model.predict(X_test)
            print(f'[Test Score] <{model_name}> - Accuracy: {accuracy_score(y_test, y_pred)}')

            print(f'Plotting Confusion Matrix for <{model_name}>')

            plt.figure()
            cm = confusion_matrix(y_test, y_pred)
            # TP FP
            # FN TN
            disp = ConfusionMatrixDisplay(cm)

            disp.plot(cmap='inferno')

            plt.title(f'Confusion Matrix - {model_name}')
            plt.show()

            if accuracy_score(y_test, y_pred) > best_accuracy:
                best_accuracy = accuracy_score(y_test, y_pred)
                best_model_evaluation = model
                best_model_name_evaluation = model_name

    return best_model_name_evaluation, best_model_evaluation, best_accuracy


# noinspection PyPep8Naming
def make_submission(submission_X: pd.DataFrame, model: GridSearchCV):
    predictions = model.predict(submission_X)
    predictions_df = pd.DataFrame(predictions)
    predictions_df.index += 1
    predictions_df.to_csv("submission.csv", header=['Incidents'], index_label='RowId')


# Main
train_df = pd.read_csv(TRAINING_DATASET_SOURCE)
test_df = pd.read_csv(TEST_DATASET_SOURCE)

train_df = pre_data_preparation(train_df)
print('Data Preprocessing done')

print('Models actived:', models_actived)
prep_test_df = data_preprocessing(test_df)
print('Preprocessed test dataset shape:', prep_test_df.shape)

train_X, test_X, train_y, test_y = data_splitting(train_df)
print('Data splitting done')
print('Data prepared for training')

print('Data shapes:', train_X.shape, test_X.shape, train_y.shape, test_y.shape)

model_training(train_X, train_y)
print('Model Training done')

best_model_name, best_model, best_acc = model_evaluation(test_X, test_y)

print(f'Best Model: {best_model_name} - Accuracy: {best_acc}')

if best_model_name is not None:
    print(f'Making submission for {best_model_name} with test accuracy of {best_acc}')
    make_submission(prep_test_df, best_model)
else:
    print("No model selected")
