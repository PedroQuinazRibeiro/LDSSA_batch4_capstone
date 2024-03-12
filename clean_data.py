from lib.data_cleaning import *

file_in = 'data/test_prod_set_all.csv'
file_out = 'data/test_prod_set_all_raw.csv'

# Read data

df = pd.read_csv(file_in)
df = preprocess_data_prod(df,filter_outcome_updated=False)
df.to_csv(file_out)
