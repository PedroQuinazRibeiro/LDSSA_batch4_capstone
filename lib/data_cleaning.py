import numpy as np
import pandas as pd

from cat_def import *


# TODO undersampling and oversampling of race and gender, use weights


def check_observation_values_validity(observation, known_categories: dict):
    ok = True

    observation_row = observation.iloc[0]

    for column in known_categories.keys():
        mapping = known_categories[column]
        good_observation = observation_row[column] in mapping.get('values')
        if good_observation:
            continue
        if mapping.get('default'):
            observation.loc[:, column] = mapping.get('default')
        else:
            ok = False
    return ok, observation


def check_valid_column(observation, valid_columns):
    """
        Validates that our observation only has valid columns

        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """

    valid_columns = set(valid_columns)

    keys = set(observation.keys())

    if len(valid_columns - keys) > 0:
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error

    if len(keys - valid_columns) > 0:
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error

    return True, ""


def check_date(observation):
    month = observation.get("month")
    day = observation.get("day")
    hour = observation.get("hour")

    if day.values[0] < 1 or day.values[0] > 31:
        error = "Field day {} is not between 1 and 31".format(day)
        return False, error

    if month.values[0] < 1 or month.values[0] > 12:
        error = "Field month {} is not between 1 and 12".format(month)
        return False, error

    if hour.values[0] < 0 or hour.values[0] > 23:
        error = "Field hour {} is not between 0 and 23".format(hour)
        return False, error

    return True, ""


def check_coordinates(observation):
    latitude = observation.get("Latitude")
    longitude = observation.get("Longitude")

    if latitude.values[0] < 48 or latitude.values[0] > 59:
        error = "Field latitude {} is not between 48 and 59".format(day)
        return False, error

    if longitude.values[0] < -9 or longitude.values[0] > 3:
        error = "Field longitude {} is not between -9 and 3".format(month)
        return False, error

    return True, ""


def resolve_unexpected_values_in_categories(df: pd.DataFrame, known_categories: dict):
    df_ = df.copy()
    defaults = 0
    dropped = 0

    for column in known_categories.keys():
        mapping = known_categories[column]
        bad_rows = df_[~df_[column].isin(mapping.get('values'))].index
        if bad_rows.empty:
            continue
        if mapping.get('default'):
            df_.iloc[bad_rows, df.columns.get_loc(column)] = mapping.get('default')
            defaults += len(bad_rows)
        else:
            df_.drop(bad_rows, inplace=True)
            dropped += len(bad_rows)

    return df_, defaults, dropped


def filter_valid_range(df: pd.DataFrame, valid_ranges: dict):
    df_ = df.copy()
    for feature, valid_range in valid_ranges.items():
        if valid_range['min'] != -np.Infinity:
            index_min = df_[df_[feature] < valid_range['min']].index
            df_.drop(index_min, inplace=True)
        if valid_range['max'] != np.Infinity:
            index_max = df_[df_[feature] > valid_range['max']].index
            df_.drop(index_max, inplace=True)
    return df_


def normalize_string(df: pd.DataFrame, columns: list):
    df_ = df.copy()
    for column in columns:
        df_[column] = df_[column].apply(lambda x: str(x).lower())
        df_[column] = df_[column].str.strip()
    return df_


def transform_date(df: pd.DataFrame, column_date: str = ''):
    df_ = df.copy()
    if str != '':
        date = pd.to_datetime(df_[column_date], infer_datetime_format=True)
        df_['year'] = date.dt.year
        df_['month'] = date.dt.month
        df_['day'] = date.dt.day
        df_['day_of_week'] = date.dt.weekday
        df_['hour'] = date.dt.hour
        df_['minute'] = date.dt.minute
    return df_


def transform_df(df: pd.DataFrame, columns_to_drop: list, columns_string: list, column_date: str):
    df_ = df.copy()
    df_ = (df_ \
           .pipe(transform_date, column_date) \
           .pipe(normalize_string, columns_string) \
           .drop(columns=columns_to_drop, errors='ignore'))

    return df_


def handle_nan_analysis(df: pd.DataFrame):
    df_ = df.copy()
    df_['Outcome linked to object of search'] = df_['Outcome linked to object of search'].fillna(value=False)
    df_['Part of a policing operation'] = df_['Part of a policing operation'].fillna(value='missing')
    df_.loc[df_['Type'] != 'vehicle search', 'Removal of more than just outer clothing'] = df_.loc[
        df_['Type'] != 'vehicle search', 'Removal of more than just outer clothing'].fillna(value=False)
    df_['Legislation'] = df_['Legislation'].fillna(value='missing')
    df_['Latitude'] = df_['Latitude'].fillna(value=0)
    df_['Longitude'] = df_['Longitude'].fillna(value=0)

    return df_


def handle_nan_model(df: pd.DataFrame):
    df_ = df.copy()

    df_['Part of a policing operation'] = df_['Part of a policing operation'].fillna(value='missing')

    df_['Latitude'] = df_['Latitude'].fillna(value=0)
    df_['Longitude'] = df_['Longitude'].fillna(value=0)

    return df_


def handle_missing_values(df: pd.DataFrame, drop: bool, fillna: dict):
    df_ = df.copy()
    if drop:
        df_ = df_.dropna()
    else:
        df_ = df_.fillna(fillna)
    return df_


def create_target(outcome):
    if outcome in outcomes_true:
        return True
    else:
        return False


def create_ethnicity(ethnicity):
    eth = 'missing'
    for key, value in ethnicities.items():
        if ethnicity in value:
            eth = key
            break

    return eth


def preprocess_data(df: pd.DataFrame, known_categories: dict):
    df_ = df.copy()

    columns_string = ['Type', 'Gender', 'Age range', 'Officer-defined ethnicity', 'Legislation',
                      'Object of search', 'station']

    column_date = 'Date'
    columns_to_drop = ['Date', 'Gender', 'Age range', 'Officer-defined ethnicity', 'year', 'minute', 'station']

    df_ = transform_df(df=df_, columns_to_drop=columns_to_drop, columns_string=columns_string, column_date=column_date)

    return df_


def preprocess_data_analysis(df: pd.DataFrame):
    df_ = df.copy()

    df_ = handle_nan_analysis(df_)

    df_['target'] = (df_['Outcome'].apply(lambda x: create_target(x))) & (
            df_['Outcome linked to object of search'] == True)

    df_['Outcome_true'] = (df_['Outcome'].apply(lambda x: create_target(x)))

    df_['ethnicity'] = df_['Self-defined ethnicity'].apply(lambda x: create_ethnicity(x))

    columns_string = ['Type', 'Gender', 'Age range', 'Officer-defined ethnicity', 'Legislation',
                      'Object of search', 'station', 'ethnicity']

    column_date = 'Date'
    columns_to_drop = ['Date', 'Self-defined ethnicity', 'Outcome']

    df_ = transform_df(df=df_, columns_to_drop=columns_to_drop, columns_string=columns_string, column_date=column_date)

    # metropolitan misses data for outcome linked to object of search
    df_ = df_[df_["station"] != 'metropolitan']

    return df_


def preprocess_data_prod(df: pd.DataFrame, filter_outcome_updated: bool = True):
    df_ = df.copy()
    df_ = df_[df_["valid"]]

    if filter_outcome_updated:
        df_ = df_[df_["outcome_updated"]]

    df_ = handle_nan_model(df_)

    columns_string = ['Type', 'Gender', 'Age range', 'Officer-defined ethnicity', 'Legislation',
                      'Object of search', 'station']

    column_date = 'Date'
    columns_to_drop = ['Date']

    df_ = transform_df(df=df_, columns_to_drop=columns_to_drop, columns_string=columns_string, column_date=column_date)

    return df_


def filter_values(df: pd.DataFrame, column_name: str, threshold: int):
    value_counts = df[column_name].value_counts()
    to_keep = value_counts[value_counts > threshold].index
    filtered = df[df[column_name].isin(to_keep)]
    return filtered
