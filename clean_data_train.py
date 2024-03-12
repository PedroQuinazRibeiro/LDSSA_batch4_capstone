from lib.data_cleaning import *

file_in = 'data/train.csv'
file_out = 'data/train_cleaned.csv'

# Read data

df = pd.read_csv(file_in)
df = preprocess_data_analysis(df)
df.to_csv(file_out)
