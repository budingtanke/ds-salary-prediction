#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:03:37 2020

@author: Han
"""

import glassdoor_scraper as gs
import pandas as pd


PATH = "/Users/linhan/Desktop/Han/Learning/Python/Softwares/chromedriver"

#This line will open a new chrome window and start the scraping.
df = gs.get_jobs("data scientist", 1000, False, PATH, 10)
df.to_csv('./data/glassdoor_raw.csv', index=False)