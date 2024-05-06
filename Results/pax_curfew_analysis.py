import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (20, 18)

df = pd.read_csv('Results/article_results.csv')

df['miss_diff'] = df.init_mis_con - df.final_mis_con
df['curfew_diff'] = df.init_curfew - df.final_curfew
df['miss_negative'] = df['miss_diff'].apply(lambda x: 0 if x >= 0 else -x)
df['curfew_negative'] = df['curfew_diff'].apply(lambda x: 0 if x >= 0 else -x)

df_mincost = df[df.model == 'mincost']
df_nnbound = df[df.model == 'nnbound']
df_udpp = df[df.model == 'udpp']



df_mincost[df_mincost.airline =='total'].miss_diff.sum()
df_mincost[df_mincost.airline =='total'].curfew_diff.sum()

df_nnbound[df_nnbound.airline =='total'].miss_diff.sum()
df_nnbound[df_nnbound.airline =='total'].curfew_diff.sum()

df_udpp[df_udpp.airline =='total'].miss_diff.sum()
df_udpp[df_udpp.airline =='total'].curfew_diff.sum()

df_mincost[df_mincost.airline =='total'].miss_negative.sum()
df_mincost[df_mincost.airline =='total'].curfew_negative.sum()

df_nnbound[df_nnbound.airline =='total'].miss_negative.sum()
df_nnbound[df_nnbound.airline =='total'].curfew_negative.sum()

df_udpp[df_udpp.airline =='total'].miss_negative.sum()
df_udpp[df_udpp.airline =='total'].curfew_negative.sum()


bins = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 204]
clusters = ['1','2','3','4','5-9','10-19','20-29','30-39','40-49', '50-74','75-99','100->']

def get_mean_std(df, bins_, item):
    mean = [df[(bins_[i] <= df["num flights"]) & (df["num flights"] < bins_[i + 1])][item].mean()
                      for i in range(len(bins_) - 1)]
    std = [df[(bins_[i] <= df["num flights"]) & (df["num flights"] < bins_[i + 1])][item].std()
                     for i in range(len(bins_) - 1)]
    return mean, std

def get_abs_vals(df, bins_, item):
    vals = [df[(bins_[i] <= df["num flights"]) & (df["num flights"] < bins_[i + 1])][item].sum()
                      for i in range(len(bins_) - 1)]
    return vals


# abs missedconnection avoided

v_mincost = get_abs_vals(df_mincost, bins, 'miss_diff')
v_nnb = get_abs_vals(df_nnbound, bins, 'miss_diff')
v_udpp = get_abs_vals(df_udpp, bins, 'miss_diff')


plt.bar(np.array(range(len(v_mincost))) - .2, v_mincost, width=.2)
plt.bar(np.array(range(len(v_nnb))), v_nnb, width=.2)
plt.bar(np.array(range(len(v_udpp))) + .2, v_udpp, width=.2)
plt.xticks(range(len(v_mincost)), clusters)
plt.ticklabel_format(style='plain', axis='y')
plt.title("PAX AVOIDED MISSEDCONNECTION PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("MISSEDCONNECTION AVOIDED")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# abs missedconnection caused

v_mincost = get_abs_vals(df_mincost, bins, 'miss_negative')
v_nnb = get_abs_vals(df_nnbound, bins, 'miss_negative')
v_udpp = get_abs_vals(df_udpp, bins, 'miss_negative')


plt.bar(np.array(range(len(v_mincost))) - .2, v_mincost, width=.2)
plt.bar(np.array(range(len(v_nnb))), v_nnb, width=.2)
plt.bar(np.array(range(len(v_udpp))) + .2, v_udpp, width=.2)
plt.xticks(range(len(v_mincost)), clusters)
plt.ticklabel_format(style='plain', axis='y')
plt.title("PAX CAUSED MISSEDCONNECTION PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("MISSEDCONNECTION CAUSED")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# abs curfew avoided

v_mincost = get_abs_vals(df_mincost, bins, 'curfew_diff')
v_nnb = get_abs_vals(df_nnbound, bins, 'curfew_diff')
v_udpp = get_abs_vals(df_udpp, bins, 'curfew_diff')

plt.bar(np.array(range(len(v_mincost))) - .2, v_mincost, width=.2)
plt.bar(np.array(range(len(v_nnb))), v_nnb, width=.2)
plt.bar(np.array(range(len(v_udpp))) + .2, v_udpp, width=.2)
plt.xticks(range(len(v_mincost)), clusters)
plt.ticklabel_format(style='plain', axis='y')
plt.title("AVOIDED CURFEW HITS PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("CURFEW HITS AVOIDED")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# abs curfew caused

v_mincost = get_abs_vals(df_mincost, bins, 'curfew_negative')
v_nnb = get_abs_vals(df_nnbound, bins, 'curfew_negative')
v_udpp = get_abs_vals(df_udpp, bins, 'curfew_negative')


plt.bar(np.array(range(len(v_mincost))) - .2, v_mincost, width=.2)
plt.bar(np.array(range(len(v_nnb))), v_nnb, width=.2)
plt.bar(np.array(range(len(v_udpp))) + .2, v_udpp, width=.2)
plt.xticks(range(len(v_mincost)), clusters)
plt.ticklabel_format(style='plain', axis='y')
plt.title("CAUSED CURFEW HITS PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("CURFEW HITS CAUSED")
plt.grid(axis="y")
plt.tight_layout()
plt.show()