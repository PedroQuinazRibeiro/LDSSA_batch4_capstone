from lib.model_utils import *

file_in = 'data/test_prod_all.csv'
file_out = 'data/test_prod_set_all.csv'

df = pd.read_csv(file_in)

df_extra = df.loc[:,['predicted_outcome','outcome','outcome_updated','proba','valid']]

def extract_features_from_saved_data(df):
    values = []
    for i in range(df["observation"].shape[0]):
        data=df["observation"].iloc[i].strip('[').strip(']').split("'")
        data1 = [ x for x in data if x!= '\n ' and x!= '' and x!= ' ']
        data2 = data1[3].split()
        data_lat = data2[1]
        data_long = data2[2]
        data_part_of_op=data2[0]
        data1.pop(3)
        data1.append(data_lat)
        data1.append(data_long)
        data1.append(data_part_of_op)
        values.append(data1)

    return values


data = extract_features_from_saved_data(df)

data1= pd.DataFrame(data,columns =["observation_id","Type","Date","Gender","Age range","Officer-defined ethnicity","Legislation","Object of search","station","Latitude","Longitude","Part of a policing operation"])

df_cat = pd.concat([data1,df_extra],axis=1)
df_cat =df_cat.set_index("observation_id")

# TODO What is object linked to object of search ?

df_cat.to_csv(file_out)