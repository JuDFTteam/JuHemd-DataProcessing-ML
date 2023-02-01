#This script uses processed JuHemd data to train ML-Models to make qualitative and quantitative predictions of a test Tc set.
#Different ML models are compared using a GridSearch algorithm, CV scores, test and train scores and - in the classification case - accuracies.
#Mutliple figures are generated for the purpose of displaying model performance and properties of the data.
#- Robin Hilgers, 2022-2023
#Published on Zenodo
#--------------------------------------------------------------------------------
# Copyright (c) 2023 Peter Grünberg Institut, Forschungszentrum Jülich, Germany
# This file is available as free software under the conditions
# of the MIT license as expressed in the LICENSE file in more detail.
#--------------------------------------------------------------------------------
#Deactivate warnings which arise regulary in the GridSearch due to unconerged models.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#Imports
#Models/HPO
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor as KNN
#Data Treatment/Splitting/Handling
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#Metrics
from sklearn.metrics import r2_score, f1_score,accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
#Plotting/Display imports/config
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = ['Arial']
from skmisc.loess import loess
#Feature Imp
import shap

#Set random seed for reproducible results
randomSeed=31415

#Enter path for data to be loaded
path=input('Enter valid path to files with Heuslers Data (Descriptors.txt ,Tc.txt, Data.txt and DataClearedFromDTF.txt containing folder): ')

#Load Data
try:
    tc = np.genfromtxt(path+'/Tc.txt')
    data = np.genfromtxt(path+'/Data.txt')[:, 1:]
    descr=np.genfromtxt(path+'/Descriptors.txt',dtype=str,delimiter=',',comments='$')[1:].astype('U100')
    descrNoDFT=np.genfromtxt(path+'/DescriptorsClearedFromDFT.txt',dtype=str,delimiter=',',comments='$')[1:].astype('U100')
    dataNoDFT=np.genfromtxt(path+'/DataClearedFromDTF.txt')[:, 1:]
    print('Path Found')
except:
    print('Path not found. This will cause an error.')

#Inp data, output data cleared of meaningless features with 0 Variance and returns removed indices.
def meaninglessFeaturesRemoval(data):
  indices=np.zeros(0,dtype=int)
  for i in range (0,len(data[0,:])):
    if np.all(data[:,i]==data[0,i]): indices=np.append(indices,int(i))
  return np.delete(data,obj=indices,axis=1), indices

#Outlier removal
ind=np.where(tc>1400.0)
data=np.delete(data,obj=ind, axis=0)
dataNoDFT=np.delete(dataNoDFT,obj=ind, axis=0)
tcORIG=tc
tc=np.delete(tc,ind)

##Remove meaningless columns
#For DFT data
dataOrig=data
descrOrig=descr
data,indices=meaninglessFeaturesRemoval(data)
print('Number of zero variance descriptors which are removed: %i'%int(len(indices)))
descr=np.delete(descr,obj= indices,axis=0).astype('U100')
#For non DFT data
dataNoDFT,indices=meaninglessFeaturesRemoval(dataNoDFT)

#Split and shuffle data
trainData,testData,trainTc,testTc=train_test_split(data,tc,test_size=0.2,shuffle=True ,random_state=randomSeed)

#Scaling
scaler=StandardScaler()
scaler.fit(trainData)
trainData=scaler.transform(trainData)
testData=scaler.transform(testData)

#Small functions for easy evaluation of regression models
def modelEvalReg(model,nameModel,ytest,ytrain,xtest,xtrain,data,tc):
    model.fit(xtrain,ytrain)
    cv=cross_val_score(model,trainData,trainTc,cv=5,scoring='r2')
    testScore=r2_score(ytest,model.predict(xtest))
    trainScore=r2_score(ytrain,model.predict(xtrain))
    print('Model performance for '+ nameModel + ' CV: ' +  str(np.round(np.mean(cv),5)) + ' R2-Test: ' + str(np.round(testScore,5)) + ' R2-Train: ' + str(np.round(trainScore ,5)))
    return

#Evaluation function for classification models
def modelEvalClass(model,nameModel,ytest,ytrain,xtest,xtrain,data,tc):
    model.fit(xtrain,ytrain)
    cv=cross_val_score(model,xtrain,ytrain.astype(int),cv=5,scoring='f1')
    testScore=f1_score(ytest.astype(int),model.predict(xtest).astype(int))
    trainScore=f1_score(ytrain.astype(int),model.predict(xtrain).astype(int))
    print('Model performance for '+ nameModel + ' CV: '+ str(np.round(np.mean(cv),5))+' F1-Test: '+ str(np.round(testScore,5))+ ' F1-Train: ' + str(np.round(trainScore,5)) + ' Test Acc.: ' + str(np.round(accuracy_score(ytest, model.predict(xtest)),5)))
    return

#Classif Func
def classify(dataTc,thresholdArray,labels):
    if len(thresholdArray)==len(labels)+1: return 4
    res=np.zeros(len(dataTc),dtype=str)
    thresholdArray=np.sort(thresholdArray)
    for i in range (0,len(dataTc)):
        for k in range (0,len(thresholdArray)):
            if dataTc[i]<thresholdArray[k]:
                res[i]=labels[k][0]
                break
        if res[i]=='':
            if dataTc[i]>thresholdArray[-1]:
                res[i]=labels[-1]
    return res

#Evaluation function for indirect classification
def modelEvalIndClass(model,nameModel,ytest,ytrain,xtest,xtrain,data,tc):
    labels = ['0', '1']  # Low Tc and High Tc Classific.
    thres = [273.15 + 200]
    model.fit(xtrain,ytrain)
    testScore=f1_score(classify(ytest,thres,labels).astype(int),classify(model.predict(xtest),thres,labels).astype(int))
    trainScore=f1_score(classify(ytrain,thres,labels).astype(int),classify(model.predict(xtrain),thres,labels).astype(int))
    print('Model performance for '+ nameModel + ' F1-Test: '+ str(np.round(testScore,5)) + ' F1-Train: '+ str(np.round(trainScore,5)) + ' Test Acc.:' + str(np.round(accuracy_score(classify(ytest,thres,labels).astype(int), classify(model.predict(xtest),thres,labels).astype(int)),5)) )
    return

#Function to automate optimization and selection
def opt(model, scor,data,tc,params):
    search = GridSearchCV(model, params, cv=5, scoring=scor)
    search.fit(data, tc)
    return search.best_estimator_

##With DFT Data
#Gradient Boosting
params={'n_estimators': [10,100,1000,10000],'learning_rate':[0.01,0.05,0.1,0.15,0.2],'loss' : ['squared_error']}
best=opt(GradientBoostingRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'GradientBoostingReg',testTc,trainTc,testData,trainData,data,tc)

#Extra Trees
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(ExtraTreesRegressor(),'r2',trainData,trainTc,params)
ETR=best#Save for ind. Class later
modelEvalReg(best,'ExtraTreesReg',testTc,trainTc,testData,trainData,data,tc)

#LassoLars
best=linear_model.LassoLarsCV(max_iter=100000,  cv=5, max_n_alphas=100000, eps=1e-16, copy_X=True)
LL=best#Save for ind. Class later
LL.fit(trainData,trainTc)
modelEvalReg(best,'LassoLarsReg',testTc,trainTc,testData,trainData,data,tc)

#Lasso
best=linear_model.LassoCV(eps=0.00001, n_alphas=1000, precompute='auto', max_iter=10000, tol=0.00001,cv=5)
reg=best#Save for Feature Importance
modelEvalReg(best,'LassoReg',testTc,trainTc,testData,trainData,data,tc)

#LinReg
best=LinearRegression()
modelEvalReg(best,'LinReg',testTc,trainTc,testData,trainData,data,tc)

#DTR
params={'criterion' : ['squared_error'], 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'DecisionTreeReg',testTc,trainTc,testData,trainData,data,tc)

#KNN
params={'n_neighbors':[2,3,4,5,6,10,15,20,25,30,40,50],'p':[1,2,3,4],'leaf_size': [5,10,20,30,40,50,60,70]}
best=opt(KNN(),'r2',trainData,trainTc,params)
modelEvalReg(best,'KNN',testTc,trainTc,testData,trainData,data,tc)

#RTR
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(RandomForestRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'RandomTreesReg',testTc,trainTc,testData,trainData,data,tc)

##Move To classification
#Prerequisites
labels=['0','1'] # Low Tc and High Tc Classific.
thres=[273.15+200]
testTcClass= classify(testTc,thres,labels).astype(int)
trainTcClass= classify(trainTc,thres,labels).astype(int)
TcClass= classify(tc,thres,labels).astype(int)

#ETC
params={'n_estimators': [10000,10000,10000,10000,10000],'criterion': ['gini']}
best=opt(ExtraTreesClassifier(),'f1',trainData,trainTcClass,params)
ETC=best
modelEvalClass(best,'ExtraTressClass',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#DTC
params={ 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'DecisionTreeClass',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#LogReg
params={'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.1,0.5,1],'solver': ['liblinear','saga']}
best=opt(LogisticRegression(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'LogisticReg',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#Ind LassoLars
modelEvalIndClass(LL,'IndirectLASSOLarsClass',testTc,trainTc,testData,trainData,data,tc)

#Ind ETR
modelEvalIndClass(ETR,'IndirectETRClass',testTc,trainTc,testData,trainData,data,tc)

##Figure generation
#testTc ETR
pred=ETR.predict(testData)
f, ax = plt.subplots(figsize=(5, 5))
df = pd.DataFrame({"Predicted $T_c$ in Kelvin":pred})
df["Test $T_c$ in Kelvin"]=pd.Series(testTc)
sns.set_theme(style="whitegrid")
g1=sns.jointplot(data=df,x="Test $T_c$ in Kelvin",y="Predicted $T_c$ in Kelvin",kind='reg',dropna = True,space = 0)
g1.ax_joint.plot([0,np.max(testTc)], [0,np.max(testTc)], 'r-', linewidth = 2)
xstd=np.std(testTc)
xmean=np.mean(testTc)
ystd=np.std(pred)
ymean=np.mean(pred)
g1.ax_marg_x.axvline(xmean, color='red', ls='--')
g1.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
g1.ax_marg_y.axhline(ymean, color='red', ls='--')
g1.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.savefig("ETR.png",dpi=1200)
plt.clf()
print('ETR test predcition plot generated.')

#ResidualPlot
sns.set_theme(style="whitegrid")
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc)/testTc)))
plt.scatter(x=testTc, y=(pred-testTc)/testTc,color="b")
l = loess(x_z,y_z)
l.fit()
pred = l.predict(x_z, stderror=True)
conf = pred.confidence()
lowess = pred.values
ll = conf.lower
ul = conf.upper
plt.plot(x_z, lowess,color='r')
plt.fill_between(x_z,ll,ul,alpha=.33)
plt.xlabel("Test $T_c$ in Kelvin")
plt.rcParams.update({'axes.labelsize':15})
plt.ylabel("$\\displaystyle\\frac{T_c^{Pred}-T_c}{T_c}$",fontsize=15)
plt.tight_layout()
plt.savefig("ResidETR.png",dpi=1200)
plt.clf()
print('ETR residues plot generated.')
plt.rcParams.update({'axes.labelsize':10})

#Mabs + Tc plot

sns.set_theme(style='whitegrid')
h=sns.jointplot(x=data[:,np.where(descr=='mAbs')[0][0]],y=tc,space = 0)
h.set_axis_labels('$\\displaystyle M_{Abs}$','$\\displaystyle T_c$')
xstd=np.std(data[:,np.where(descr=='mAbs')[0][0]])
xmean=np.mean(data[:,np.where(descr=='mAbs')[0][0]])
ystd=np.std(tc)
ymean=np.mean(tc)
h.ax_marg_x.axvline(xmean, color='red', ls='--')
h.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
h.ax_marg_y.axhline(ymean, color='red', ls='--')
h.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
plt.savefig('TcvsM.png',dpi=600)
plt.clf()
print('T_c vs M_abs plot generated.')

#Test Tc LASSOLars
pred=LL.predict(testData)
f, ax = plt.subplots(figsize=(5, 5))
df = pd.DataFrame({"Predicted $T_c$ in Kelvin":pred})
df["Test $T_c$ in Kelvin"]=pd.Series(testTc)
sns.set_theme(style="whitegrid")
g1=sns.jointplot(data=df,x="Test $T_c$ in Kelvin",y="Predicted $T_c$ in Kelvin",kind='reg',dropna = True,space = 0)
g1.ax_joint.plot([0,np.max(testTc)], [0,np.max(testTc)], 'r-', linewidth = 2)
xstd=np.std(testTc)
xmean=np.mean(testTc)
ystd=np.std(pred)
ymean=np.mean(pred)
g1.ax_marg_x.axvline(xmean, color='red', ls='--')
g1.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
g1.ax_marg_y.axhline(ymean, color='red', ls='--')
g1.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.savefig("LL.png",dpi=1200)
plt.clf()
print('LASSOLars test plot generated.')

#Histogram of Atomic #
dataOccup=data[:,np.where(descr=='Atomic Numbers of Atom 1')[0][0]:np.where(descr=='Atomic Numbers of Atom 1')[0][0]+4]
dataOccup=pd.DataFrame(data=dataOccup, columns=['Site 1', 'Site 2', 'Site 3', 'Site 4'])
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(10, 5))
sns.despine(f)
sns.histplot(
    data=dataOccup, 
    multiple="stack",
    palette="light:m_r",
discrete=True,
    edgecolor=".3",
    linewidth=.5,
    log_scale=False,
    bins=int(np.max(data[:,np.where(descr=='Atomic Numbers of Atom 1')[0][0]:np.where(descr=='Atomic Numbers of Atom 1')[0][0]+4])),
)
ax.set(xlabel='Atomic Number')
plt.grid()
sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
plt.savefig("Hist.png",dpi=1200)
plt.clf()
print('Histogram generated.')

##Feature Importance using SHAP
#Clean Descr Array
for i in range (0,len(descr)):
    descr[i]=descr[i].replace('Density Param for atom # 27','Cobalt Density')
    descr[i]=descr[i].replace('#','\#')
    descr[i]=descr[i].replace('atom','Atom')
    descr[i]=descr[i].replace('_',' ')
    descr[i]=descr[i].replace('mAbs','$|M|$')
    descr[i]=descr[i].replace('AbsM1','$|M_1|$')
    descr[i]=descr[i].replace('mTot','$M$')
    descr[i]=descr[i].replace('M1','$M_1$')
    descr[i]=descr[i].replace('AbsM3','$|M_3|$')
    descr[i]=descr[i].replace('M3','$M_3$')
    descr[i]=descr[i].replace('Magn State','Mangetic State')

#SHAP Beeswarm Plot (Mean SHAP Values, SHAP Values and SHAP Values for best 9 Descr.)
X=pd.DataFrame(trainData,columns=descr)
explainer=shap.TreeExplainer(ETR,X)
shap_values=explainer(X)
shap.plots.bar(shap_values,show=False)
plt.xlabel('Mean SHAP Value')
plt.tight_layout()
plt.savefig('MeanSHAP.png',dpi=1200)
plt.clf()
shap.summary_plot(shap_values,show=False)
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('TotSHAP.png',dpi=1200)
plt.clf()

shap.summary_plot(shap_values,show=False,max_display=9)
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('RedSHAP.png',dpi=1200)
plt.clf()
print('Feature importance plot generated.')

#Histogram of Tcs
sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 5))
sns.despine(f)
yHist1=pd.DataFrame.from_records([testTc,tcORIG],['Test set $T_c$','$T_c$'])
yHist1=yHist1.transpose()
sns.histplot(
    data=yHist1,
    multiple="layer",
    edgecolor=".3",
    linewidth=.5,binwidth=100
)
ax.set(xlabel='$T_c$ in Kelvin')
sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
plt.savefig('TcHist.png',dpi=1200)
plt.clf()
print('Tcs Histogram generated.')

##Confusion Matrices
#Indirect via ETR
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(ETR.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.savefig('ConfIndETR.png')
plt.clf()

#Indirect via LASSOLars
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(LL.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.savefig('ConfIndLL.png')

#Direct via ETC
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),ETC.predict(testData).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.savefig('ConfDirETC.png')

##Without DFT
#Shuffle and split data
trainData,testData,trainTc,testTc=train_test_split(dataNoDFT,tc,test_size=0.2,shuffle=True ,random_state=randomSeed)

#Scaling 
scalerNoDFT=StandardScaler()
scalerNoDFT.fit(trainData)
trainData=scalerNoDFT.transform(trainData)
testData=scalerNoDFT.transform(testData)
data=dataNoDFT

#Gradient Boosting
params={'n_estimators': [10,100,1000,10000],'learning_rate':[0.01,0.05,0.1,0.15,0.2],'loss' : ['squared_error']}
best=opt(GradientBoostingRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'GradientBoostingReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

#Extra Trees
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(ExtraTreesRegressor(),'r2',trainData,trainTc,params)
ETR=best#Save for ind. Class later
modelEvalReg(best,'ExtraTreesReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

##Feature Importance using SHAP
#Clean Descr Array
for i in range (0,len(descrNoDFT)):
    descrNoDFT[i]=descrNoDFT[i].replace('Density Param for atom # 27','Cobalt Density')
    descrNoDFT[i]=descrNoDFT[i].replace('#','\#')
    descrNoDFT[i]=descrNoDFT[i].replace('atom','Atom')
    descrNoDFT[i]=descrdescrNoDFT[i].replace('_',' ')

#SHAP Beeswarm Plot (Mean SHAP Values, SHAP Values and SHAP Values for best 9 Descr.)
X=pd.DataFrame(trainData,columns=descrNoDFT)
explainer=shap.TreeExplainer(ETR,X)
shap_values=explainer(X)
shap.plots.bar(shap_values,show=False)
plt.xlabel('Mean SHAP Value')
plt.tight_layout()
plt.savefig('MeanSHAP.png',dpi=1200)
plt.clf()
shap.summary_plot(shap_values,show=False)
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('TotSHAP.png',dpi=1200)
plt.clf()

shap.summary_plot(shap_values,show=False,max_display=9)
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('RedSHAP.png',dpi=1200)
plt.clf()
print('Feature importance plot for non-DFT Features generated.')



#LassoLars
best=linear_model.LassoLarsCV( max_iter=100000,  cv=5, max_n_alphas=100000, eps=1e-16, copy_X=True)
LL=best#Save for ind. Class later
LL.fit(trainData,trainTc)
modelEvalReg(best,'LassoLarsReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

#Lasso
best=linear_model.LassoCV(eps=0.00001, n_alphas=1000, precompute='auto', max_iter=10000, tol=0.00001,cv=5)
modelEvalReg(best,'LassoReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

#LinReg
best=LinearRegression()
modelEvalReg(best,'LinReg',testTc,trainTc,testData,trainData,data,tc)

#DTR
params={'criterion' : ['squared_error'], 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'DecisionTreeReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

#KNN
params={'n_neighbors':[2,3,4,5,6,10,15],'p':[1,2,3,4],'leaf_size': [5,10,20,30,40,50,60,70]}
best=opt(KNN(),'r2',trainData,trainTc,params)
modelEvalReg(best,'KNN no DFT data',testTc,trainTc,testData,trainData,data,tc)

#RTR
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(RandomForestRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'RandomTreesReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

##Move To classification
#Prerequisites
labels=['0','1'] # Low Tc and High Tc Classific.
thres=[273.15+200]
testTcClass= classify(testTc,thres,labels).astype(int)
trainTcClass= classify(trainTc,thres,labels).astype(int)
TcClass= classify(tc,thres,labels).astype(int)

#ETC
params={'n_estimators': [10000,10000,10000,10000,10000],'criterion': ['gini']}
best=opt(ExtraTreesClassifier(),'f1',trainData,trainTcClass,params)
ETC=best# For confu Matrix later
modelEvalClass(best,'ExtraTreesClass no DFT data',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#DTC
params={ 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'DescisionTreesClass no DFT data',testTcClass.astype(int),trainTcClass.astype(int),testData,trainData,data,TcClass.astype(int))

#LogReg
params={'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.1,0.5,1],'solver': ['liblinear','saga']}
best=opt(LogisticRegression(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'Logistic no DFT data',testTcClass.astype(int),trainTcClass.astype(int),testData,trainData,data,TcClass.astype(int))

#Ind LassoLars
modelEvalIndClass(LL,'IndirectLASSOLarsClass no DFT data',testTc,trainTc,testData,trainData,data,tc)

#Ind ETR
modelEvalIndClass(ETR,'IndirectETR Class no DFT data',testTc,trainTc,testData,trainData,data,tc)

##Confusion Matrices
#Indirect via ETR
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(ETR.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.savefig('ConfIndETRNoDFT.png')
plt.clf()

#Indirect via LASSOLars
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(LL.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.savefig('ConfIndLLNoDFT.png')

#Direct via ETC
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),ETC.predict(testData).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.savefig('ConfDirETCNoDFT.png')
