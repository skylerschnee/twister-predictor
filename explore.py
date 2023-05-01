#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('figure', figsize=(13, 7))
plt.rc('font', size=16)



def get_monthly_avg_vis(y):
    '''
    show_cas_by_month takes in y, a df of the traget variable, then displays a 
    visualization of that variable's mean by month    
    '''
    ax = y.groupby(y.index.month).mean().plot.bar(width=.9, color='cyan', ec='black')
    plt.xticks(rotation=45, ha='right')
    ax.set(title='Average Casualties by Month', xlabel='Month', ylabel='Casualties')

    # Display the total for each bar
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', labels=[f'{int(j.get_height())}' for j in i])
    
def show_cas_over_time(train_by_year, test_by_year):
    '''
    show_cas_over_time takes in train and test subsets of tornado data and 
    will show a visualization of casualties due to tornados over time
    '''
    # Plot train and test
    plt.plot(train_by_year.index, train_by_year.casualties)
    plt.plot(test_by_year.index, test_by_year.casualties)
    # Title, label, legend
    plt.title('Tornado Casualties over Time')
    plt.ylabel('Amount of Casualties')
    plt.xlabel('Year')
    plt.legend(['Train','Test'])
    plt.show()
    

def get_cas_decade_vis(df_by_decade):
    '''
    
    '''
    subset = df_by_decade[['injuries', 'fatalities']]

    ax = subset.plot(kind='bar', stacked=True, width=0.8, edgecolor='black')

    # Set the axis labels and title
    ax.set_xlabel('Decade')
    ax.set_ylabel('Casualties')
    ax.set_title('Casualties by Decade')
    # Format the x-tick labels to display the decade range
    labels = [f'{decade}-{decade+9}' for decade in subset.index.year]
    ax.set_xticklabels(labels)

    # Show the plot
    plt.show()

