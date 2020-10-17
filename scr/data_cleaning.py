#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:59:04 2020

@author: Han
"""

import pandas as pd

df = pd.read_csv('../data/glassdoor_raw.csv')

# =============================================================================
# Job Title             object
# Salary Estimate       object
# Job Description       object
# Rating               float64
# Company Name          object
# Location              object
# Headquarters           int64
# Size                  object
# Founded                int64
# Type of ownership     object
# Industry              object
# Sector                object
# Revenue               object
# Competitors            int64

# salary parsing
# company name text only
# location get state
# age of company
# parsing job description
# job title
# size
# revenue
# length of job description
# =============================================================================
print(df.columns)
print(df.dtypes)

################# salary #####################

# remove records withouth salary
df = df[df['Salary Estimate'] != '-1']

# remove text in salary
df['if_hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['if_employer'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer' in x.lower() else 0)
salary = df['Salary Estimate'].apply(lambda x: x.replace('(Glassdoor est.)', '').replace('\n','').replace('$','').replace('K',''))
salary = salary.apply(lambda x: x.lower().replace('per hour','').replace('(employer est.)','').replace('employer provided salary:','').strip())

# get avarage salary
min_salary = salary.apply(lambda x: int(x.split('-')[0]))
max_salary = salary.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (min_salary + max_salary)/2

# convert hourly salary to yearly
df['avg_salary'] = df.apply(lambda x: x['avg_salary'] * 2000/1000 if x['if_hourly'] == 1 else x['avg_salary'], axis = 1)


################ company name ###############

# remove rating
df['company_text'] = df.apply(lambda x: x['Company Name'][:-4] if (x['Rating'] != -1) else x['Company Name'], axis=1)


################ location get state ###############
df['state'] = df['Location'].apply(lambda x: x.split(',')[1].strip())
df['state'].value_counts()
df.loc[df['state']=='Los Angeles', 'state'] = 'CA'

################ age of company ###############
df['age'] = df['Founded'].apply(lambda x: x if x < 0 else 2020-x)

################ parsing job description ###############
#print(df['Job Description'][1])

# programming languages
df['sql'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['java'] = df['Job Description'].apply(lambda x: 1 if 'java' in x.lower() else 0)
df['sas'] = df['Job Description'].apply(lambda x: 1 if 'sas' in x.lower() else 0)
df['matlab'] = df['Job Description'].apply(lambda x: 1 if 'matlab' in x.lower() else 0)
df['javascript'] = df['Job Description'].apply(lambda x: 1 if 'javascript' in x.lower() else 0)
df['c++'] = df['Job Description'].apply(lambda x: 1 if 'c++' in x.lower() else 0)
df['scala'] = df['Job Description'].apply(lambda x: 1 if 'scala' in x.lower() else 0)

# big data
df['hadoop'] = df['Job Description'].apply(lambda x: 1 if 'hadoop' in x.lower() else 0)
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['hive'] = df['Job Description'].apply(lambda x: 1 if 'hive' in x.lower() else 0)

# deep learning
df['deep_learning'] = df['Job Description'].apply(lambda x: 1 if 'deep learning' in x.lower() else 0)
df['nlp'] = df['Job Description'].apply(lambda x: 1 if ('nlp' in x.lower()) or ('natural language' in x.lower())  else 0)
df['cv'] = df['Job Description'].apply(lambda x: 1 if ('cv2' in x.lower()) or ('computer vision' in x.lower())  else 0)
df['tensorflow'] = df['Job Description'].apply(lambda x: 1 if 'tensorflow' in x.lower() else 0)
df['pytorch'] = df['Job Description'].apply(lambda x: 1 if 'pytorch' in x.lower() else 0)
df['keras'] = df['Job Description'].apply(lambda x: 1 if 'keras' in x.lower() else 0)


################ job title ###############
def title_simplifier(title):
    if ('scientist' in title.lower()) or ('data science' in title.lower()):
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'data analyst' 
    elif ('machine learning' in title.lower()) or ('mle' in title.lower()):
        return 'machine learning engineer'
    else:
        return 'others'


def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
        return 'senior'
    elif 'jr' in title.lower() or 'junior' in title.lower():
        return 'junior'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    elif 'lead' in title.lower():
        return 'lead'
    else:
        return 'others'
    
df['title_simple'] = df['Job Title'].apply(title_simplifier)
df['seniority'] = df['Job Title'].apply(seniority)

df['title_simple'].value_counts()
df['seniority'].value_counts()


################ Type of ownership ####################
df['Type of ownership'].value_counts()
df['ownership'] = df['Type of ownership'].apply(lambda x: x if x in ['Company - Public', 'Company - Private', 'Subsidiary or Business Segment', 'Nonprofit Organization', 'Government'] else 'other')
df['ownership'].value_counts()


################ Industry ####################
df['Industry'].value_counts()
df['industry'] = df['Industry'].apply(lambda x: x if x in ['Biotech & Pharmaceuticals', 'Aerospace & Defense', 'Internet', 'Computer Hardware & Software', 'IT Services', 'Health Care Services & Hospitals', 'Consulting', 'Enterprise Software & Network Solutions', 'Advertising & Marketing', 'Federal Agencies', 'Banks & Credit Unions', 'Publishing', 'Insurance Carriers'] else 'others')

################### Sector ###################
df['Sector'].value_counts()
df['sector'] = df['Sector'].apply(lambda x: x if x in ['Biotech & Pharmaceuticals', 'Information Technology', 'Business Services', 'Aerospace & Defense', 'Finance', 'Health Care', 'Manufacturing', 'Insurance', 'Media', 'Retail', 'Government', 'Education', 'Oil, Gas, Energy & Utilities'] else 'others')

################### size ###################
df['size'] = df['Size'].apply(lambda x: x.replace('Employees', '').strip())

################### revenue ###################
df['revenue'] = df['Revenue'].apply(lambda x: x.replace('(USD)', '').strip())

#################### job description ##################
df['description'] = df['Job Description'].apply(lambda x: len(x))

df_cleaned = df[['Rating', 'size', 'ownership', 'industry', 'sector', 'revenue', 'avg_salary', 'company_text', 'state','age', 'sql', 'python', 'java', 'sas', 'matlab', 'javascript', 'c++', 'scala', 'hadoop', 'spark', 'hive', 'deep_learning', 'nlp', 'cv', 'tensorflow', 'pytorch', 'keras', 'title_simple', 'seniority', 'description']]

df_cleaned.to_csv('../data/glassdoor_cleaned.csv', index=False)


