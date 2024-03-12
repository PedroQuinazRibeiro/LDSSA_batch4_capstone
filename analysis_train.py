from lib.model_utils import *



file_in = 'data/train_cleaned.csv'

df = pd.read_csv(file_in)
df_train = df

target="target"
y_test=df_train[target]

discrimination_precision = DiscriminationObservationRate(X_test=df_train.drop(columns=[target]), y_true=y_test, feature='station',
      sensitive_column='Officer-defined ethnicity')

discrimination_precision.analyze()
df_precision_group = discrimination_precision.fill_dict()

metrics = metrics_discrimination(df_precision_group)
print(metrics)

file_out = 'data/train_discrimation_precision.csv'
df_precision_group.to_csv(file_out)

