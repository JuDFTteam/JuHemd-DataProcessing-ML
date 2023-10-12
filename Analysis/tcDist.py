#Visualization of the Tc's
#- Robin Hilgers, 2022-2023
#Published on Zenodo
#--------------------------------------------------------------------------------
# Copyright (c) 2023 Peter Grünberg Institut, Forschungszentrum Jülich, Germany
# This file is available as free software under the conditions
# of the MIT license as expressed in the LICENSE file in more detail.
#--------------------------------------------------------------------------------

import seaborn as sns
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = ['Arial']
import pandas as pd
path=input('Enter valid path to files with Heuslers Data (Tc.txt containing folder): ')
#Load Data
try:
    tc = np.genfromtxt(path+'/Tc.txt')
    print('Path Found')
except:
    print('Path not found. This will cause an error.')


sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 5))
sns.despine(f)
yHist1=pd.DataFrame.from_records([tc],['$T_c$'])
yHist1=yHist1.transpose()
sns.histplot(
    data=yHist1,
    palette="Blues",
#    multiple="layer",
    edgecolor=".3",
    linewidth=.5,binwidth=100
)
legend=ax.get_legend().remove()

#plt.legend(show=False)
ax.set(xlabel='$T_c$ in Kelvin')
sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
plt.tight_layout()
plt.savefig('NoTestTcHist.png',dpi=1200)
plt.clf()

print('Tcs Histogram generated.')

