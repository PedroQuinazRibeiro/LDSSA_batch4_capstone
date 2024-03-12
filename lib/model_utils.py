# Imports
import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_crosstab(df: pd.DataFrame, factor1, factor2):
    crosstab = pd.crosstab(df[factor1], df[factor2])
    print(crosstab)
    print('\n')
    return crosstab


def search_likelihood(df: pd.DataFrame, factor):
    normalization_census_2011 = {'White': 0.86, 'Black': 0.033, 'Other': 0.01, 'Asian': 0.075, 'Mixed': 0.022}
    normalization_population = pd.Series(normalization_census_2011).sort_index()
    relative_search_frequency = df[factor].value_counts(normalize=True, dropna=True).sort_index()
    likelihood = np.multiply(1.0 / normalization_population, relative_search_frequency)
    normalized_likehood = likelihood / (likelihood.min())
    print(normalized_likehood)
    print('\n')
    return normalized_likehood


def feature_preprocessing(df: pd.DataFrame, target_column: str, ignore_columns: list):
    all_features = [feature for feature in df.columns if
                    feature != target_column and feature != 'Unnamed: 0' and feature not in ignore_columns]

    categorical_features = list(
        df[all_features].select_dtypes(include=['object']).columns)

    numerical_features = [feature for feature in all_features if feature not in categorical_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', ce.one_hot.OneHotEncoder(use_cat_names=True, handle_unknown='indicator'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    return all_features, categorical_features, numerical_features, preprocessor


def feature_importance(pipeline: Pipeline, categorical_features: list, numerical_features: list, X_train: pd.DataFrame):
    categorical_encode_step = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    classifier_step = pipeline.named_steps['classifier']

    onehot_columns = categorical_encode_step.get_feature_names()
    importances = pd.Series(data=classifier_step.feature_importances_,
                            index=np.array(numerical_features + list(onehot_columns)))

    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, importances.index[indices[f]], importances[indices[f]]))


def explain_regression(pipeline: Pipeline, features):
    classifier_step = pipeline.named_steps['classifier']
    betas = classifier_step.coef_[0]
    indices = np.argsort(np.abs(betas))
    print("Regression coefficient ranking")
    for i in range(0, len(betas)):
        print('+ (%0.3f * %s)' % (betas[indices[i]], features[indices[i]]))


class DiscriminationAnalyzer():
    def __init__(self, X_test, y_true, feature='station', sensitive_column='ethnicity', min_samples=30):
        self.X_test = X_test
        self.y_true = y_true
        self.feature = feature
        self.min_samples = min_samples
        self.sensitive_column = sensitive_column

        self.departments = X_test[feature].unique()
        self.sensitive_classes = X_test[sensitive_column].unique()

        self.column_names = [str(cat) for cat in self.sensitive_classes]
        self.sucess_rate_dict = {}

        if sensitive_column == 'Officer-defined ethnicity':
            self.index_norm_group = np.where(self.sensitive_classes == 'white')[0][0]
        elif sensitive_column == 'Gender':
            self.index_norm_group = np.where(self.sensitive_classes == 'male')[0][0]
        elif sensitive_column == 'Object of search':
            self.index_norm_group = np.where(self.sensitive_classes == 'controlled drugs')[0][0]
        if sensitive_column == 'Age range':
            self.index_norm_group = np.where(self.sensitive_classes == 'over 34')[0][0]

    def analyze(self):
        pass

    def fill_dict(self):
        df_sucess_rate_group = pd.DataFrame().from_dict(self.sucess_rate_dict, orient="index",
                                                        columns=self.column_names)

        if self.sensitive_column == 'Officer-defined ethnicity':
            df_sucess_rate_group = df_sucess_rate_group.drop(columns=['white'])
        elif self.sensitive_column == 'Gender':
            df_sucess_rate_group = df_sucess_rate_group.drop(columns=['male'])
        elif self.sensitive_column == 'Object of search':
            df_sucess_rate_group = df_sucess_rate_group.drop(columns=['controlled drugs'])
        elif self.sensitive_column == 'Age range':
            df_sucess_rate_group = df_sucess_rate_group.drop(columns=['Age range'])

        return df_sucess_rate_group


class DiscriminationObservationRate(DiscriminationAnalyzer):
    def __init__(self, X_test, y_true, feature='station', sensitive_column='ethnicity', min_samples=30):
        super().__init__(X_test, y_true, feature, sensitive_column, min_samples)

    def analyze(self):
        for department in self.departments:
            rate_per_group = []
            for sensitive_class in self.sensitive_classes:
                mask_all_outcomes = (self.X_test[self.sensitive_column] == sensitive_class) & (
                            self.X_test[self.feature] == department)

                counts_per_group = np.sum(mask_all_outcomes)

                mask_outcome_true = (mask_all_outcomes) & (self.y_true)
                n_samples_true = np.sum(mask_outcome_true)

                if counts_per_group > self.min_samples:
                    rate_per_group.append(n_samples_true / counts_per_group)

                else:
                    rate_per_group.append(np.nan)

            if not np.isnan(rate_per_group[self.index_norm_group]):
                self.sucess_rate_dict[department] = rate_per_group - rate_per_group[self.index_norm_group]


class DiscriminationModelPrecision(DiscriminationAnalyzer):
    def __init__(self, X_test, y_true, y_pred, feature='station', sensitive_column='ethnicity', min_samples=30):
        super().__init__(X_test, y_true, feature, sensitive_column, min_samples)
        self.y_pred = y_pred

    def analyze(self):
        for department in self.departments:
            precision_per_group = []
            for cat in self.sensitive_classes:
                mask = (self.X_test[self.sensitive_column] == cat) & (
                        self.X_test[self.sensitive_column] != 'missing') & (
                               self.X_test[self.feature] == department)
                if mask.sum() > self.min_samples:
                    precision_per_group.append(precision_score(self.y_true[mask], self.y_pred[mask]))
                else:
                    precision_per_group.append(np.nan)

            if not np.isnan(precision_per_group[self.index_norm_group]):
                self.sucess_rate_dict[department] = precision_per_group - precision_per_group[self.index_norm_group]


def compute_discrimination(X_test, y_true, y_pred, sensitive_column='ethnicity', min_samples=30):
    sensitive_classes = X_test[sensitive_column].unique()
    discrimination = {}
    for sensitive_class in sensitive_classes:
        mask = X_test[sensitive_column] == sensitive_class
        discrimination[sensitive_class] = precision_score(y_true[mask], y_pred[mask])

    return discrimination


def metrics_discrimination(df: pd.DataFrame):
    df_ = df.copy()
    df_ = df_.fillna(0)
    df_station_with_discriminate = df_[(df_.abs() > 0.05).any(axis=1)]
    df_station_pos_discrimination = df_[(df_ > 0.05).any(axis=1)]
    df_station_neg_discrimination = df_[(df_ < -0.05).any(axis=1)]
    df_station_non_black_asian_discrimination = df_.drop(columns=["black", "asian"])
    df_station_mixed_other_discrimination = df_station_non_black_asian_discrimination[
        (df_station_non_black_asian_discrimination.abs() > 0.05).any(axis=1)]
    median_discrimination_stations = df_[(df_.abs() > 0.05).any(axis=1)].sum(axis=1).median()
    spread_discrimination_stations = df_[(df_.abs() > 0.05).any(axis=1)].sum(axis=1).max() - df_[
        (df_.abs() > 0.05).any(axis=1)].sum(axis=1).min()

    out_dict = {}
    out_dict["discriminate"] = df_station_with_discriminate.shape[0]
    out_dict["discriminate_pos"] = df_station_pos_discrimination.shape[0]
    out_dict["discriminate_neg"] = df_station_neg_discrimination.shape[0]
    out_dict["discriminate_mixed_other"] = df_station_mixed_other_discrimination.shape[0]
    out_dict["median_discrimination"] = median_discrimination_stations
    out_dict["spread_discrimination"] = spread_discrimination_stations

    return out_dict
