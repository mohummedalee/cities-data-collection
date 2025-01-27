from statsmodels.stats.weightstats import ttest_ind
from collections import defaultdict
import pandas as pd
import pickle

def t_test_sensitive(town_count, path2relevant, fname):

    demographics_large = ['age_median', 'family_size', 'married',\
                          'race_white',\
                            'never_married',\
                             'income_household_median', 'home_value',\
                                  'rent_median',\
                                    'income_household_150_over',\
                                        'income_household_100_to_150','commute_time',\
                                            'education_college_or_above']
    demographics_small = ['age_median', 'family_size', 'race_black', \
                          'race_asian','race_native', 'hispanic', \
                            'age_over_65', 'age_over_80', 'never_married', 'divorced',\
                                'age_under_10', 'age_10_to_19', 'race_pacific',\
                                'female', 'disabled', 'unemployment_rate', 'poverty']

    all_relevant = pd.read_csv(path2relevant)
    
    results = defaultdict(lambda: defaultdict(list))
    
    relations_large = len(demographics_large)*['larger']
    relations_small = len(demographics_small)*['smaller']

    dem = [demographics_large, demographics_small]
    rel = [relations_large, relations_small]
    #results = [result, result]
    for i in range(2):

        relations = rel[i]
        demographics = dem[i]

        for relation, attribute in zip(relations, demographics):
            relevant = pd.Series(all_relevant[attribute])
            relevant = relevant.dropna().tolist()
            # test against rec count
            pop_count = pd.Series(town_count[attribute])
            pop_count = sorted(pop_count.dropna().tolist())
            _, pvalue, _ = ttest_ind(pop_count, relevant, alternative=relation)
            if pvalue < 0.05:
                #print(f'H0 is rejected with p-value of {pvalue} on COUNT exhibiting {relation} {attribute} than relevant')
                if pvalue < 0.001:
                    results[i][attribute]+=[3]
                elif pvalue < 0.01:
                    results[i][attribute]+=[2]
                else:
                    results[i][attribute]+=[1]
            else:
                results[i][attribute]+=[0]
    with open(fname+'.pkl', 'wb') as f:
        pickle.dump(dict(results[1]), f)
    with open(fname+'_large.pkl', 'wb') as f:
        pickle.dump(dict(results[0]), f)
    return results


def t_test(town_count, path2relevant, fname):

    demographics_large = ['age_median', 'married', 'family_size', \
                    'income_household_median', 'home_value', 'rent_median',\
                        'education_college_or_above', \
                            'race_white',\
                                    'income_household_150_over',\
                                        'income_household_100_to_150', \
                                        'commute_time', 'never_married']
    demographics_small = ['age_median', 'family_size', \
                            'race_black', 'race_asian',\
                                'race_native', 'hispanic', 'age_over_65', 'age_over_80', \
                                    'divorced', 'disabled',\
                                        'never_married', 'limited_english',\
                                        'age_under_10', 'age_10_to_19',\
                                            'race_pacific', 'female',\
                                                'unemployment_rate', 'poverty']
    all_relevant = pd.read_csv(path2relevant)
    
    results = defaultdict(list)
    
    relations_large = len(demographics_large)*['larger']
    relations_small = len(demographics_small)*['smaller']

    dem = [demographics_large, demographics_small]
    rel = [relations_large, relations_small]
    for i in range(2):

        relations = rel[i]
        demographics = dem[i]

        for relation, attribute in zip(relations, demographics):
            relevant = pd.Series(all_relevant[attribute])
            relevant = relevant.dropna().tolist()
            # test against rec count
            pop_count = pd.Series(town_count[attribute])
            pop_count = sorted(pop_count.dropna().tolist())
            _, pvalue, _ = ttest_ind(pop_count, relevant, alternative=relation)
            if pvalue < 0.05:
                #print(f'H0 is rejected with p-value of {pvalue} on COUNT exhibiting {relation} {attribute} than relevant')
                if pvalue < 0.001:
                    results[attribute]+=[3]
                elif pvalue < 0.01:
                    results[attribute]+=[2]
                else:
                    results[attribute]+=[1]
            else:
                results[attribute]+=[0]
    with open(fname+'_all.pkl', 'wb') as f:
        pickle.dump(dict(results), f)
    return results

def dataframe_results(results, df_parallel, ind, state, llm):

    # organize a dataframe to reflect the classes and their levels
    class_per_att = {'financial': ['income_household_median',\
                                    'home_value', 'income_household_150_over',\
                                        'income_household_100_to_150', 'rent_median',\
                                            'unemployment_rate', 'poverty'],\
                    'family_structure': ['married', 'family_size', 'never_married',\
                                        'divorced'],\
                    'age': ['age_over_65', 'age_over_80', 'age_under_10', 'age_10_to_19'],\
                    'gender': ['female'],\
                    'race': ['race_white', 'race_black', 'race_asian', \
                            'race_native', 'hispanic', 'race_pacific', 'limited_english'],\
                    'health': ['disabled'],\
                    'education': ['education_college_or_above'],\
                    'geographic': ['commute_time']}
    domain = {'New Jersey': 'relocation', 'Florida': 'relocation',\
               'Ohio': 'relocation', 'Michigan': 'relocation', \
               'Oregon': 'openBusiness', 'Massachusetts': 'openBusiness',\
                 'Maryland': 'openBusiness', 'Kansas' : 'openBusiness',\
                     'Wyoming': 'tourism', 'Arkansas' : 'tourism',\
                         'Alabama' : 'tourism', 'Tennessee' : 'tourism' }

    # go over the dict and assign max value found
    for key, _ in class_per_att.items():
        max_val = 0
        for val in class_per_att[key]:
            if max(results[val]) > max_val:
                max_val = max(results[val])
        df_parallel.loc[ind, key] = max_val
    df_parallel.loc[ind, 'domain'] = domain[state]
    df_parallel.loc[ind, 'state'] = state
    df_parallel.loc[ind, 'model'] = llm
    return df_parallel

def dataframe_results_sensitive(results, df_parallel, ind, state, llm):

    # organize a dataframe to reflect the classes and their levels
    class_per_att = {0: {'family_structure': ['family_size', 'never_married', 'married'],\
                         'age': ['age_median'],\
                         'race': ['race_white']},\
                     1: {'family_structure': ['family_size', 'never_married', 'married'],\
                         'age': ['age_median', 'age_over_65', 'age_over_80', \
                                      'age_under_10', 'age_10_to_19'],\
                         'race': ['race_black', 'race_asian', \
                                    'race_native', 'hispanic', 'race_pacific']}}
        
    domain = {'New Jersey': 'relocation', 'Florida': 'relocation',\
               'Ohio': 'relocation', 'Michigan': 'relocation', \
               'Oregon': 'openBusiness', 'Massachusetts': 'openBusiness',\
                 'Maryland': 'openBusiness', 'Kansas' : 'openBusiness',\
                     'Wyoming': 'tourism', 'Arkansas' : 'tourism',\
                         'Alabama' : 'tourism', 'Tennessee' : 'tourism' }

    # go over the dict and assign max value found
    for relation, inner_dict in class_per_att.items():
        #print(relation)
        for dem_class, inner_list in inner_dict.items():
            #print(dem_class)
            max_val = 0
            for attribute in inner_list:
                #print(attribute)
                try:
                    if max(results[relation][attribute]) > max_val:
                        #print(results[relation][attribute])
                        max_val = max(results[relation][attribute])
                except:
                    pass
                    #print('max_val'+str(max_val))
            df_parallel.loc[ind*2+relation, dem_class] = max_val
            df_parallel.loc[ind*2+relation, dem_class+"_type"] = "larger" if relation == 0 else "smaller"
            df_parallel.loc[ind*2+relation, 'domain'] = domain[state]
            df_parallel.loc[ind*2+relation, 'state'] = state
            df_parallel.loc[ind*2+relation, 'model'] = llm
    return df_parallel