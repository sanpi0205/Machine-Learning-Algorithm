#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:12:21 2017

@author: zhangbo
"""

import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

def parse(x):
  return datetime.strptime(x, '%Y %m %d %H')

dataset = pd.read_csv('air_pollution_data.csv',
                      parse_dates=[['year','month','day','hour']],
                      index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
dataset.columns = ['polution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

dataset = dataset[24:]
dataset['polution'].fillna(0, inplace=True)

dataset.to_csv('pollution.csv')

values = dataset.values

groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
plt.figure()
for group in groups:
  plt.subplot(len(groups), 1, i)
  plt.plot(values[:, group])
  plt.title(dataset.columns[group], y=0.5, loc='right')
  i += 1
plt.show()






  