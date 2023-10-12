#Script visualizing the distribution of symmetries within JuHemd.
#- Robin Hilgers, 2023
#Published on Zenodo
#--------------------------------------------------------------------------------
# Copyright (c) 2023 Peter Grünberg Institut, Forschungszentrum Jülich, Germany
# This file is available as free software under the conditions
# of the MIT license as expressed in the LICENSE file in more detail.
#--------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = ['Arial']

#Enter path for data to be loaded
path=input('Enter valid path to files with Heuslers Data (sym.txt containing folder): ')

#Load Data
try:
    df = pd.read_csv(path+'/sym.txt', sep=", ", header=None, names=["Symmetry"])
    print('Path Found')
except:
    print('Path not found. This will cause an error.')

dfFracDis=pd.DataFrame()
#Drop fractional structures
dfFracDis["Frac"]=df["Symmetry"].str.contains('=')
print(dfFracDis)
print(dfFracDis["Frac"].value_counts()[True])

#Count major symmetry types
df=df.loc[dfFracDis["Frac"]==False].dropna()
coutL21=df["Symmetry"].str.contains('L21_').value_counts()[True]
coutY=df["Symmetry"].str.contains('Y_').value_counts()[True]
coutC1b=df["Symmetry"].str.contains('C1b_').value_counts()[True]
coutD03=df["Symmetry"].str.contains('D03_').value_counts()[True]
coutX=df["Symmetry"].str.contains('X_').value_counts()[True]
countA2=df["Symmetry"].str.contains('A2_').value_counts()[True]
countB2=df["Symmetry"].str.contains('B2_').value_counts()[True]

print(df["Symmetry"].str.contains('').value_counts()[True])
print(coutL21)
total=776
other=total-coutL21-coutY-coutC1b-coutD03-coutX-countA2-countB2
print(other)
#Visualization
dfCounts = pd.DataFrame([coutL21,coutY,coutC1b,coutD03,coutX,countA2,countB2,other], columns=['Structure Count'], index=["$\mathrm{L2}_1$","Y","$\mathrm{C1}_b$","$\mathrm{D0}_3$","XA","A2","B2","Other"])
dfCounts=dfCounts.sort_values(by=["Structure Count"],axis=0,ascending=False)
print(dfCounts)
sns.set_theme(style="whitegrid")
sns.barplot(data=dfCounts,y="Structure Count",x=dfCounts.index,  edgecolor=".3",
    linewidth=.5,color=sns.color_palette("Blues")[2])
plt.tight_layout()
plt.savefig("juHemdSymGroups.png",dpi=1200)
