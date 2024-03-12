from lib.model_utils import *



file_in = 'data/test_prod_set_all_cleaned.csv'
file_out = 'data/analysis.csv'

df = pd.read_csv(file_in)
df_test = df[df["outcome_updated"]]

target="outcome"
y_test=df_test[target]
y_pred_=df_test["predicted_outcome"]

discrimination_precision = DiscriminationModelPrecision(X_test=df_test.drop(columns=[target]), y_true=y_test, y_pred=y_pred_, feature='station',
      sensitive_column='Officer-defined ethnicity')
discrimination_precision.analyze()
df_precision_group = discrimination_precision.fill_dict()

metrics = metrics_discrimination(df_precision_group)
print(metrics)

file_out = 'data/prod_discrimation_precision.csv'
df_precision_group.to_csv(file_out)

