#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:59:04 2020

@author: linhan
"""

import pandas as pd

df = pd.read_csv('glassdoor_raw.csv')

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
# =============================================================================
print(df.columns)
print(df.dtypes)

################# salary #####################

df = df[df['Salary Estimate'] != '-1']

df['if_hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['if_employer'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer' in x.lower() else 0)
salary = df['Salary Estimate'].apply(lambda x: x.replace('(Glassdoor est.)', '').replace('\n','').replace('$','').replace('K',''))
salary = salary.apply(lambda x: x.lower().replace('per hour','').replace('(employer est.)','').replace('employer provided salary:','').strip())

min_salary = salary.apply(lambda x: int(x.split('-')[0]))
max_salary = salary.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (min_salary + max_salary)/2

################ company name ###############

df['company_text'] = df.apply(lambda x: x['Company Name'][:-4] if (x['Rating'] != -1) else x['Company Name'], axis=1)


################ location get state ###############
df['state'] = df['Location'].apply(lambda x: x.split(',')[1].strip())
df['state'].value_counts()

################ age of company ###############
df['age'] = df['Founded'].apply(lambda x: x if x < 0 else 2020-x)

################ parsing job description ###############
print(df['Job Description'][0])

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


df.to_csv('glassdoor_cleaned.csv', index=False)


