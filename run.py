import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
from core.app import Core


methods = ('KMeans', 'DBSCAN', 'AFF')
features = ('age', 'job', 'martial', 'education', 'default', 'balance', 'housing', 'loan', 'duration')
plt.rc('font', size=16)

#set style of plots
sns.set_style('white')

#define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)

df = st.cache(pd.read_csv)("data/bank-full.csv")
st.title("Bank clustering")
is_check = st.sidebar.checkbox("Display Data")

method = st.sidebar.radio('Which cluster method?', methods)
xaxis = st.sidebar.radio('Feature on x axis', features)
yaxis = st.sidebar.radio('Feature on y axis', features)

cluster = st.sidebar.button("Cluster!")

if is_check:
	st.write(df)

if cluster:
	core = Core()
	core.cluster(method)
	results = core.get_results(xaxis, yaxis)

	#plot data with seaborn
	ax = sns.lmplot(data=results, x='x', y='y', hue='label', fit_reg=False, legend=True, legend_out=True)
	ax.set(xlabel=xaxis, ylabel=yaxis)
	st.pyplot()
