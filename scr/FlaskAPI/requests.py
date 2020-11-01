#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:39:57 2020

@author: Han
"""

import requests
import pandas as pd
import numpy as np
import pickle

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}

# load test value, which is a list of features
with open('models/test_value.p', 'rb') as f:
    data_input = pickle.load(f)
    
data = {"input": data_input}
r = requests.get(URL,headers=headers, json=data)

r.json()
