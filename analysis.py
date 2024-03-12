from lib.model_utils import *



file_in = 'data/train_cleaned.csv'
file_out = ''

df = pd.read_csv(file_in)


df = df.replace({'missing': np.nan})



cross_tab_ethnicity = make_crosstab(df,'target','ethnicity')
cross_tab_ethnicity1 = make_crosstab(df,'target','Officer-defined ethnicity')


cross_tab_ethnicity2 = make_crosstab(df,'Officer-defined ethnicity','ethnicity')
cross_tab_gender = make_crosstab(df,'target','Gender')
cross_tab_age = make_crosstab(df,'target','Age range')
cross_tab_cloth = make_crosstab(df[df['Gender']=='female'],'Removal of more than just outer clothing','ethnicity')
cross_tab_cloth = make_crosstab(df[df['Gender']=='female'],'Removal of more than just outer clothing','Age range')
cross_tab_linked_to_outcome = make_crosstab(df[df["station"]!="metropolitan"],'target','Outcome linked to object of search')
cross_tab_2 = make_crosstab(df,'Outcome linked to object of search','ethnicity')

cross_tab_station = make_crosstab(df,'ethnicity','station')
cross_tab_object = make_crosstab(df,'ethnicity','Object of search',)


frequency_ethnicity = search_likelihood(df,'ethnicity')



