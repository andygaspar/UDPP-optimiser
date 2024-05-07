import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (20, 18)

mincost_color = 'lightgray'
nnbound_color = 'black'
udpp_color = 'gray'

df = pd.read_csv('Results/iterative_test_results.csv')

df['miss_diff'] = df.init_mis_con - df.final_mis_con
df['curfew_diff'] = df.init_curfew - df.final_curfew
df['miss_negative'] = df['miss_diff'].apply(lambda x: 0 if x >= 0 else -x)
df['curfew_negative'] = df['curfew_diff'].apply(lambda x: 0 if x >= 0 else -x)

df_mincost = df[df.model == 'mincost']
df_nnbound = df[df.model == 'nnbound']
df_udpp = df[df.model == 'udpp']


df_total = df[df.airline=='total']
