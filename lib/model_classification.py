# Imports
import json
import pickle

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import data_cleaning
import model_utils


def train_and_predict(df: pd.DataFrame, station: str = '', object_of_search: str = '', ethnicity:str = ''):
    df_ = df[df["station"] != 'metropolitan']


    target = 'target'

    sensitive_columns = [ 'year', 'minute',
                         "Outcome linked to object of search",'Outcome_true','ethnicity',
                        'weight_class','eth_gender',
                          'station',
                          'Officer-defined ethnicity','Gender',
                          'Age range',
                          'Removal of more than just outer clothing','preds_proba',
                          'observation_id']



    features, categorical_features, numerical_features, preprocessor = model_utils.feature_preprocessing(df, target,
                                                                                                         ignore_columns=sensitive_columns)

    print("Numerical features {}".format(numerical_features))
    print("Categorical features {}".format(categorical_features))

    if station != '' or object_of_search != '':
        df_ = df_[df_["station"] == station]

    if object_of_search != '':
        df_ = df_[df_["Object of search"] == object_of_search]



    df_train, df_test = train_test_split(df_, test_size=0.3,
                                         random_state=42, stratify=df_['target'])

    X_train = df_train[features]
    y_train = df_train[target]

    pipeline = Pipeline(
        [('preprocessor', preprocessor),
     ('classifier',RandomForestClassifier(min_samples_leaf=30, class_weight="balanced", random_state=42, n_jobs=-1))])


    pipeline.fit(X_train, y_train)

    X_train.loc[:, 'preds_proba'] = pipeline.predict_proba(X_train)[:, 1]

    X_test = df_test[features]
    y_test = df_test[target]



    X_test.loc[:,'preds_proba'] = pipeline.predict_proba(X_test)[:, 1]


    try:
        model_utils.feature_importance(pipeline, categorical_features, numerical_features, X_train)
    except ValueError:
        pass

    precision, recall, thresholds = precision_recall_curve(y_true=y_test, probas_pred=X_test['preds_proba'])

    precision = precision[:-1]
    recall = recall[:-1]

    min_index = [i for i, prec in enumerate(precision) if prec >= 0.2][0]
    print(min_index)
    print(thresholds[min_index])
    print(precision[min_index])
    print(recall[min_index])


    y_pred_ = X_test['preds_proba'] > 0.327

    precision = precision_score(y_test, y_pred_)
    recall = recall_score(y_test, y_pred_)
    score = roc_auc_score(y_score=X_test['preds_proba'], y_true=y_test)

    print("roc_auc_score {}".format(score))
    print("precision {}".format(precision))
    print("recall {}".format(recall))


    discrimination_precision = model_utils.DiscriminationModelPrecision(X_test=df_test.drop(columns=[target]), y_true=y_test, y_pred=y_pred_, feature='station',
                                                                        sensitive_column='Officer-defined ethnicity')
    discrimination_precision.analyze()
    df_precision_group = discrimination_precision.fill_dict()


    with open('../columns.json', 'w') as fh:
        json.dump(X_train.columns.tolist(), fh)

    with open('../dtypes.pickle', 'wb') as fh:
        pickle.dump(X_train.dtypes, fh)

    joblib.dump(pipeline, '../pipeline.pickle');

    return y_test, y_pred_, X_train, X_train['preds_proba'], df_train,df_test, df_precision_group



file_in = '../data/train.csv'
file_out = ''

df = pd.read_csv(file_in)
df= data_cleaning.preprocess_data_analysis(df)



y_test = {}
y_pred = {}

stations = df['station'].unique()
objects_of_search = df['Object of search'].unique()
ethnicities = df['Officer-defined ethnicity'].unique()


y_test, y_pred, X_train, y_train_pred,df_train,df_test,discrimination_precision = train_and_predict(df)

file_out = '../data/model_discrimination_precision.csv'
discrimination_precision.to_csv(file_out)



metrics = model_utils.metrics_discrimination(discrimination_precision)
print(metrics)



