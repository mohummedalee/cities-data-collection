import pandas as pd
import re
from collections import Counter, defaultdict
from parallel_utility_llm import t_test, t_test_sensitive
from parallel_utility_llm import dataframe_results, dataframe_results_sensitive
import sys


prefix = path_to_llm_responses_csv_file
relevant_pre = prefix+'relevant_db/' # path to the relevant sets created for ground truth
mode = 'sg'
mode1 = 'sensitive' # sensitive indicates sensitivity to one group being larger or smaller than the other
#mode1 = ''
if mode == 'sg':
    sg_path = prefix+'code/all_models_single_constraint.csv'
else:
    sg_path = prefix+'code/all_models_generic.csv'
us_db_path = prefix+'us_cities_db/simplemaps_uscitiesv1/uscities.csv'
df_sc = pd.read_csv(sg_path)
us_db = pd.read_csv(us_db_path)

# load single constraint
df_sc = df_sc.drop(['Unnamed: 0'], axis=1)

# all unique prompts
prompts = df_sc.prompt.unique()

# all unique models
models = df_sc.model.unique()

def extract_cities_from_response(response_df, models):
    # return a dict of LLM and city list
    llm_responses = {}
    # extrat LLM type
    
    for model in models:
        model_df = response_df[response_df['model']==model]
        # extract cities
        city_pre = 'rec_city'
        cities = []
        for i in range(1,6):
            city = []
            city = model_df[city_pre+str(i)].to_list()
            cities.extend(city)
        llm_responses[model]=cities
    return llm_responses

relevant = {'Kansas': 'KN_crime_rates.csv', 'Florida': 'FL_age_over_65.csv', \
            'Ohio': 'OH_between_70k_n_20k.csv', 'Michigan': 'MI_population_proper.csv',\
                'Oregon': 'OR_20_n_30.csv', 'Wyoming': 'WY_wildlife_habitat.csv',\
                    'Alabama': 'AL_public_fishing_ponds.csv', \
                      'Tennessee': 'TN_historical_sites.csv',\
                        'Arkansas': 'AR_state_parks.csv',\
                        'New Jersey': 'NJ_bike_scores.csv', \
                        'Maryland': 'MD_walk_scores.csv', \
                        'Massachusetts': 'MA_public_transit_scores.csv'}

states = ['Kansas', 'Florida', 'Ohio', 'Michigan', 'Oregon', \
          'Wyoming', 'Alabama', 'Tennessee', 'Arkansas', 'New Jersey',\
             'Maryland', 'Massachusetts']
#states = ['Michigan', 'Florida']

# a dataframe for final results
if mode1 == 'sensitive':
    df_parallel = pd.DataFrame(columns=['domain', 'model', 'state', 'age', 'race', 'family_structure'])
else:
    df_parallel = pd.DataFrame(columns=['domain', 'model', 'state', 'age','financial',\
                                                'family_structure','gender',\
                                                'race','health','education', 'geographic'])

for j, state in enumerate(states):
    # find the state's prompt
    results = defaultdict(list)
    for i, string in enumerate(prompts):
        match = re.search(state, string)
        if match:
            prompt = string
            break
    # extract all query's data
    response_df = df_sc[df_sc['prompt']==prompt]
    llm_dict = extract_cities_from_response(response_df, models)
    # compute per query per LLM per domain

    # build demographics db from us db
    state_df = us_db[us_db['state_name']==state]
    town_count = pd.DataFrame({})
    for m, (model, cities) in enumerate(llm_dict.items()):
        for city in cities:
            town_data = state_df[state_df['city'] == city]
            town_count = pd.concat([town_count, town_data], axis=0)
        if mode1 == 'sensitive': 
            results = t_test_sensitive(town_count, relevant_pre+relevant[state], prefix+'code/rq2_2/dicts/'+state+'_'+model+'_'+mode)
            df = dataframe_results_sensitive(results, df_parallel, j*len(models)+m, state, model) # domain
        else:
            results = t_test(town_count, relevant_pre+relevant[state], prefix+'code/rq2_2/dicts/'+state+'_'+model+'_'+mode)
            df = dataframe_results(results, df_parallel, j*len(models)+m, state, model) # domain

df.to_csv(prefix+'code/rq2_2/parallel_df_model_'+mode+'_'+mode1+'.csv')
