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

#Deactivate warnings which arise regulary in the GridSearch due to unconverged models.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#Imports
#Models/HPO
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,ExtraTreesClassifier,RandomForestClassifier, GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from xgboost import XGBClassifier as XGBC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neighbors import KNeighborsClassifier as KNNClass
#Data Treatment/Splitting/Handling
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#Metrics
from sklearn.metrics import r2_score, f1_score,accuracy_score,confusion_matrix, ConfusionMatrixDisplay,mean_absolute_error,precision_score,recall_score
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
import os 
os.environ["PYTHONHASHSEED"]=str(randomSeed)
np.random.seed(42)
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
descrNoDFT=np.delete(descrNoDFT,obj= indices,axis=0).astype('U100')

#Split and shuffle data
trainData,testData,trainTc,testTc=train_test_split(data,tc,test_size=0.2,shuffle=True ,random_state=randomSeed)

#Scaling
scaler=StandardScaler()
scaler.fit(trainData)
trainData=scaler.transform(trainData)
testData=scaler.transform(testData)



#Histogram of Tcs
sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 5))
sns.despine(f)
yHist1=pd.DataFrame.from_records([tc,testTc],['Complete data set w/o Outliers','Test data set'])
yHist1=yHist1.transpose()

sns.histplot(
    data=yHist1,
    multiple="layer",
    palette=sns.color_palette("Paired",2),
    edgecolor=".3",
    linewidth=.5,binwidth=100
)
ax.set(xlabel='$T_c$ in Kelvin')

plt.tight_layout()
plt.savefig('TcHist.png',dpi=1200)
plt.clf()
print('Tcs Histogram generated.')

#Small functions for easy evaluation of regression models
def modelEvalReg(model,nameModel,ytest,ytrain,xtest,xtrain,data,tc):
    model.fit(xtrain,ytrain)
    cv=cross_val_score(model,trainData,trainTc,cv=4,scoring='r2')
    testScore=r2_score(ytest,model.predict(xtest))
    trainScore=r2_score(ytrain,model.predict(xtrain))
    print('Model performance for '+ nameModel + ' CV: ' +  str(np.round(np.mean(cv),2)) + ' R2-Test: ' + str(np.round(testScore,2)) + ' R2-Train: ' + str(np.round(trainScore ,2))+" MAE Train: "+ str(mean_absolute_error(ytrain,model.predict(xtrain)))+ " MAE Test: "+str(mean_absolute_error(ytest,model.predict(xtest))))
    return

#Evaluation function for classification models
def modelEvalClass(model,nameModel,ytest,ytrain,xtest,xtrain,data,tc):
    model.fit(xtrain,ytrain)
    cv=cross_val_score(model,xtrain,ytrain.astype(int),cv=4,scoring='f1')
    testScore=f1_score(ytest.astype(int),model.predict(xtest).astype(int))
    trainScore=f1_score(ytrain.astype(int),model.predict(xtrain).astype(int))
    print('Model performance for '+ nameModel + ' CV: '+ str(np.round(np.mean(cv),2))+' F1-Test: '+ str(np.round(testScore,2))+ ' F1-Train: ' + str(np.round(trainScore,2)) + ' Test Acc.: ' + str(np.round(accuracy_score(ytest, model.predict(xtest)),2))+' Test Precision.: ' + str(np.round(precision_score(ytest, model.predict(xtest)),2))+' Test Recall.: ' + str(np.round(recall_score(ytest, model.predict(xtest)),2)))
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
    print('Model performance for '+ nameModel + ' F1-Test: '+ str(np.round(testScore,2)) + ' F1-Train: '+ str(np.round(trainScore,2)) + ' Test Acc.:' + str(np.round(accuracy_score(classify(ytest,thres,labels).astype(int), classify(model.predict(xtest),thres,labels).astype(int)),2))+'Test Recall.:' + str(np.round(recall_score(classify(ytest,thres,labels).astype(int), classify(model.predict(xtest),thres,labels).astype(int)),2))+'Test Precision.:' + str(np.round(precision_score(classify(ytest,thres,labels).astype(int), classify(model.predict(xtest),thres,labels).astype(int)),2)) )
    return

#Function to automate optimization and selection
def opt(model, scor,data,tc,params):
    search = GridSearchCV(model, params, cv=4, scoring=scor,refit=True)
    search.fit(data,tc)
    return search.best_estimator_

##With DFT Data
#Extra Trees
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(ExtraTreesRegressor(),'r2',trainData,trainTc,params)
ETR=best#Save for ind. Class later
modelEvalReg(best,'ExtraTreesReg',testTc,trainTc,testData,trainData,data,tc)

##Feature Importance using SHAP
#Clean Descr Array

for i in range (0,len(descr)):
    descr[i]=descr[i].replace('_',' ')
    descr[i]=descr[i].replace('Density Param for atom # 27','Cobalt Density')
    descr[i]=descr[i].replace('Density Param for atom # 25','Manganese Density')
    descr[i]=descr[i].replace('Density Param for atom # 26','Iron Density')
    descr[i]=descr[i].replace('#','\#')
    descr[i]=descr[i].replace('atom','Atom')
    descr[i]=descr[i].replace('eleneg of Atom 2','$\chi^{(2)}$')
    descr[i]=descr[i].replace('elenegTOTAL','$\\chi^{\\mathrm{Tot}}$')
    descr[i]=descr[i].replace('mAbs','$M_{\\mathrm{Abs}}$')
    descr[i]=descr[i].replace('TOTAL \# of Valence electrons','Total $e^{\\mathrm{val}}$')
    descr[i]=descr[i].replace('AbsM1','$|m_1|$')
    descr[i]=descr[i].replace('mTot','$M$')
    descr[i]=descr[i].replace('M1','$m_1$')
    descr[i]=descr[i].replace('AbsM3','$|m_3|$')
    descr[i]=descr[i].replace('M3','$m_3$')
    descr[i]=descr[i].replace('Magn State','Magnetic State')

#SHAP Beeswarm Plot (Mean SHAP Values, SHAP Values and SHAP Values for best 9 Descr.)
X=pd.DataFrame(trainData,columns=descr)
explainer=shap.TreeExplainer(ETR,X)
shap_values=explainer(X)
shap.plots.bar(shap_values,show=False)
plt.xlabel('Mean SHAP Value')
plt.tight_layout()
plt.savefig('MeanSHAP.png',dpi=1200)
plt.clf()

shap.summary_plot(shap_values,show=False,plot_type="layered_violin",plot_size=(10,5))
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('TotSHAP.png',dpi=1200)
plt.clf()

shap.summary_plot(shap_values,show=False,max_display=9,plot_type="layered_violin",plot_size=(10,5))
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('RedSHAP.png',dpi=1200)
plt.clf()
print('Feature importance plot generated.')


#Gradient Boosting
params={'n_estimators': [10,100,1000,10000],'learning_rate':[0.01,0.05,0.1,0.15,0.2],'loss' : ['squared_error']}
best=opt(GradientBoostingRegressor(),'r2',trainData,trainTc,params)
GBR=best
modelEvalReg(best,'GradientBoostingReg',testTc,trainTc,testData,trainData,data,tc)


#LassoLars
best=linear_model.LassoLarsCV(max_iter=100000,  cv=4, max_n_alphas=100000, eps=1e-16, copy_X=True)
best.fit(trainData,trainTc)
modelEvalReg(best,'LassoLarsReg',testTc,trainTc,testData,trainData,data,tc)

#Lasso
best=linear_model.LassoCV(eps=0.00001, n_alphas=1000, precompute='auto', max_iter=10000, tol=0.00001,cv=4)
LL=best#Save for ind. Class later
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

#xgboost
xgBoost=XGBRegressor()
xgBoostSet={'n_estimators': [4,10,100,1000,5000],'eta':[0.01,0.05,0.1,0.2,0.3],"lambda":[1.1,1.05,1.0],"eval_metric":["mse"]}
best=opt(XGBRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'XGBost',testTc,trainTc,testData,trainData,data,tc)

##Move To classification
#Prerequisites
labels=['0','1'] # Low Tc and High Tc Classific.
thres=[273.15+200]
testTcClass= classify(testTc,thres,labels).astype(int)
trainTcClass= classify(trainTc,thres,labels).astype(int)
TcClass= classify(tc,thres,labels).astype(int)

#ETC
params={'n_estimators': [10000],'criterion': ['gini','logloss']}
best=opt(ExtraTreesClassifier(),'f1',trainData,trainTcClass,params)
ETC=best
modelEvalClass(best,'ExtraTressClass',testTcClass,trainTcClass,testData,trainData,data,TcClass)


#RF
params={'n_estimators': [10000],'criterion': ['gini','logloss']}
best=opt(RandomForestClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'RandForClass',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#XGBClass
params={'n_estimators': [4,10,100,1000,5000],'eta':[0.01,0.05,0.1,0.2,0.3],"lambda":[1.1,1.05,1.0]}
best=opt(XGBC(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'XGBClass',testTcClass,trainTcClass,testData,trainData,data,TcClass)


#GBClass
params={'n_estimators': [4,10,100,1000,5000],'learning_rate':[0.01,0.05,0.1,0.15,0.2]}
best=opt(GradientBoostingClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'GBClass',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#KNN
params={'n_neighbors':[2,3,4,5,6,10,15,20,25,30,40,50],'p':[1,2,3,4],'leaf_size': [5,10,20,30,40,50,60,70]}
best=opt(KNNClass(),'f1',trainData,trainTc,params)
modelEvalClass(best,'KNNClass NoDFT',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#DTC
params={ 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'DecisionTreeClass',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#LogReg
params={'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.1,0.5,1],'solver': ['liblinear','saga']}
best=opt(LogisticRegression(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'LogisticReg',testTcClass,trainTcClass,testData,trainData,data,TcClass)
predLogReg=best.predict(testData)
#Ind Lasso
modelEvalIndClass(LL,'IndirectLASSOClass',testTc,trainTc,testData,trainData,data,tc)

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
g1.ax_marg_x.axvline(xmean, color='red', ls=':')
g1.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
g1.ax_marg_y.axhline(ymean, color='red', ls=':')
g1.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
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
plt.plot(x_z, lowess,color='r',linestyle=":")
plt.fill_between(x_z,ll,ul,alpha=.33)
plt.xlabel("Test $T_c$ in Kelvin")
plt.rcParams.update({'axes.labelsize':15})
plt.ylabel("$\\displaystyle\\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$",fontsize=15)
plt.tight_layout()

plt.savefig("ResidETR.png",dpi=1200)
plt.clf()
print('ETR residues plot generated.')

#ResidualDistPlot
sns.set_theme(style="whitegrid")
pred=ETR.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.histplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"))
plt.xlabel("$\\displaystyle \\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$")
plt.xlim((-2,2))
plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()

plt.savefig("ResidDistETR.png",dpi=1200)
plt.clf()
print('ETR residues dist plot generated.')

plt.rcParams.update({'axes.labelsize':10})

#ResidualKDEDistPlot
sns.set_theme(style="whitegrid")
pred=ETR.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.kdeplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"),fill=True)
plt.xlabel("$\\displaystyle \\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$")
plt.xlim((-3,3))
plt.ylabel("Distribution Density")
plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()

plt.savefig("ResidKDEDistETR.png",dpi=1200)
plt.clf()
print('ETR residues KDE dist plot generated.')


#ResidualKDEDistPlot
sns.set_theme(style="whitegrid")
pred=ETR.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.kdeplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"),fill=True)
plt.xlabel("$\\displaystyle \\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$")
plt.ylabel("Distribution Density")
plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()
plt.savefig("ResidKDEDistETRNoLim.png",dpi=1200)
plt.clf()
print('ETR residues KDE dist plot generated.')


#Test Tc LASSO
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
g1.ax_marg_x.axvline(xmean, color='red', ls=':')
g1.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
g1.ax_marg_y.axhline(ymean, color='red', ls=':')
g1.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
plt.savefig("LL.png",dpi=1200)
plt.clf()
print('LASSO test plot generated.')


#ResidualPlot
sns.set_theme(style="whitegrid")
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc)/testTc)))
f, ax = plt.subplots(figsize=(5, 5))
ax.scatter(x=testTc, y=(pred-testTc)/testTc,color="b")
axin=ax.inset_axes([0.21, 0.35, 0.74, 0.3],
    xlim=(200, 1050), ylim=(-1, 1),xticklabels=[],yticklabels=[])
axin.scatter(x=testTc, y=(pred-testTc)/testTc,color="b")
l = loess(x_z,y_z)
l.fit()
pred = l.predict(x_z, stderror=True)
conf = pred.confidence()
lowess = pred.values
ll = conf.lower
ul = conf.upper
ax.plot(x_z, lowess,color='r',linestyle=":")
axin.plot(x_z, lowess,color='r',linestyle=":")
ax.indicate_inset_zoom(axin,edgecolor="black")
ax.fill_between(x_z,ll,ul,alpha=.33)
axin.fill_between(x_z,ll,ul,alpha=.33)
plt.xlabel("Test $T_c$ in Kelvin")
plt.rcParams.update({'axes.labelsize':15})
plt.ylabel("$\\displaystyle\\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$",fontsize=15)
plt.tight_layout()

plt.savefig("ResidLasso.png",dpi=1200)
plt.clf()
print('Lasso residues plot generated.')

#ResidualDistPlot
sns.set_theme(style="whitegrid")
pred=LL.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.histplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"))
plt.xlabel("$T_c^{\\mathrm{Pred}}-T_c$")


plt.xlabel("Test $T_c$ in Kelvin")
plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()

plt.savefig("ResidDistLL.png",dpi=1200)
plt.clf()
print('LL residues dist plot generated.')

plt.rcParams.update({'axes.labelsize':10})

#ResidualKDEDistPlot
sns.set_theme(style="whitegrid")
pred=LL.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.kdeplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"),fill=True)
plt.xlabel("$\\displaystyle \\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$")
plt.xlim((-3,3))
plt.ylabel("Distribution Density")
plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()

plt.savefig("ResidKDEDistLL.png",dpi=1200)
plt.clf()
print('LL residues KDE dist plot generated.')

#ResidualKDEDistPlot
f, ax = plt.subplots(figsize=(5, 5))
sns.set_theme(style="whitegrid")
pred=LL.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.kdeplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"),fill=True)
plt.xlabel("$\\displaystyle \\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$")
plt.ylabel("Distribution Density")
plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()

plt.savefig("ResidKDEDistLLNoLim.png",dpi=1200)
plt.clf()
print('LL residues KDE dist plot generated.')
#Histogram of Atomic #
dataOccup=data[:,np.where(descr=='Atomic Numbers of Atom 1')[0][0]:np.where(descr=='Atomic Numbers of Atom 1')[0][0]+4]
dataOccup=pd.DataFrame(data=dataOccup, columns=['Site 1', 'Site 2', 'Site 3', 'Site 4'])
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(10, 5))
sns.despine(f)
sns.histplot(
    data=dataOccup, 
    multiple="stack",
    palette="Blues",#light:m_r
discrete=True,
    edgecolor=".3",
    linewidth=.45,
    log_scale=False,
    bins=int(np.max(data[:,np.where(descr=='Atomic Numbers of Atom 1')[0][0]:np.where(descr=='Atomic Numbers of Atom 1')[0][0]+4])),
)
ax.set(xlabel='Atomic Number')
plt.grid()

plt.tight_layout()
plt.savefig("Hist.png",dpi=1200)
plt.clf()
print('Histogram generated.')

#Mabs + Tc plot

X=pd.DataFrame(trainData,columns=descr)
sns.set_theme(style='whitegrid')
h=sns.jointplot(x=data[:,np.where(descr=='$M_{\\mathrm{Abs}}$')[0][0]],y=tc,space = 0)
h.set_axis_labels('$\\displaystyle M_{\\mathrm{Abs}}$ in $\mu_B$','$\\displaystyle T_c$ in Kelvin')
xstd=np.std(data[:,np.where(descr=='$M_{\\mathrm{Abs}}$')[0][0]])
xmean=np.mean(data[:,np.where(descr=='$M_{\\mathrm{Abs}}$')[0][0]])
ystd=np.std(tc)
ymean=np.mean(tc)
h.ax_marg_x.axvline(xmean, color='red', ls=':')
h.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
h.ax_marg_y.axhline(ymean, color='red', ls=':')
h.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
plt.savefig('TcvsM.png',dpi=1200)
plt.clf()
print('T_c vs M_abs plot generated.')

X=pd.DataFrame(trainData,columns=descr)
sns.set_theme(style='whitegrid')
h=sns.jointplot(x=data[:,np.where(descr=='$|m_1|$')[0][0]],y=tc,space = 0)
h.set_axis_labels('$\\displaystyle |m_1|$ in $\mu_B$','$\\displaystyle T_c$ in Kelvin')
xstd=np.std(data[:,np.where(descr=='$|m_1|$')[0][0]])
xmean=np.mean(data[:,np.where(descr=='$|m_1|$')[0][0]])
ystd=np.std(tc)
ymean=np.mean(tc)
h.ax_marg_x.axvline(xmean, color='red', ls=':')
h.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
h.ax_marg_y.axhline(ymean, color='red', ls=':')
h.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
plt.savefig('TcvsAbsM1.png',dpi=1200)
plt.clf()
print('T_c vs M_abs,1 plot generated.')

X=pd.DataFrame(trainData,columns=descr)
sns.set_theme(style='whitegrid')
h=sns.jointplot(x=data[:,np.where(descr=='$m_1$')[0][0]],y=tc,space = 0)
h.set_axis_labels('$\\displaystyle m_1$ in $\mu_B$','$\\displaystyle T_c$ in Kelvin')
xstd=np.std(data[:,np.where(descr=='$m_1$')[0][0]])
xmean=np.mean(data[:,np.where(descr=='$m_1$')[0][0]])
ystd=np.std(tc)
ymean=np.mean(tc)
h.ax_marg_x.axvline(xmean, color='red', ls=':')
h.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
h.ax_marg_y.axhline(ymean, color='red', ls=':')
h.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
plt.savefig('TcvsM1.png',dpi=1200)
plt.clf()
print('T_c vs M_1 plot generated.')

X=pd.DataFrame(trainData,columns=descr)
sns.set_theme(style='whitegrid')
h=sns.jointplot(x=data[:,np.where(descr=='Total $e^{\\mathrm{val}}$')[0][0]],y=tc,space = 0)
h.set_axis_labels('Total Number of $\\displaystyle e^{\\mathrm{val}}$','$\\displaystyle T_c$ in Kelvin')
xstd=np.std(data[:,np.where(descr=='Total $e^{\\mathrm{val}}$')[0][0]])
xmean=np.mean(data[:,np.where(descr=='Total $e^{\\mathrm{val}}$')[0][0]])
ystd=np.std(tc)
ymean=np.mean(tc)
h.ax_marg_x.axvline(xmean, color='red', ls=':')
h.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
h.ax_marg_y.axhline(ymean, color='red', ls=':')
h.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
plt.savefig('TcvEvalTot.png',dpi=1200)
plt.clf()
print('T_c vs Total $e^{val}$ plot generated.')

#Mtot + Tc plot
sns.set_theme(style='whitegrid')
h=sns.jointplot(x=data[:,np.where(descr=='$M$')[0][0]],y=tc,space = 0)
h.set_axis_labels('$\\displaystyle M$ in $\mu_B$','$\\displaystyle T_c$ in Kelvin')
xstd=np.std(data[:,np.where(descr=='$M$')[0][0]])
xmean=np.mean(data[:,np.where(descr=='$M$')[0][0]])
ystd=np.std(tc)
ymean=np.mean(tc)
h.ax_marg_x.axvline(xmean, color='red', ls=':')
h.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
h.ax_marg_y.axhline(ymean, color='red', ls=':')
h.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
plt.savefig('TcvsMTot.png',dpi=1200)
plt.clf()
print('T_c vs M_tot plot generated.')


##Confusion Matrices
#Indirect via ETR
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(ETR.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.savefig('ConfIndETR.png')
plt.clf()

#LogReg predLogReg
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),predLogReg.astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.savefig('ConflogReg.png')
#Indirect via LASSO
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
print('Confusion Matrices generated.')





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
GBR=best
modelEvalReg(best,'GradientBoostingReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

#Extra Trees
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(ExtraTreesRegressor(),'r2',trainData,trainTc,params)
ETR=best#Save for ind. Class later
modelEvalReg(best,'ExtraTreesReg no DFT data',testTc,trainTc,testData,trainData,data,tc)


#testTc ETR
pred=GBR.predict(testData)
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
g1.ax_marg_x.axvline(xmean, color='red', ls=':')
g1.ax_marg_x.axvspan(xmean - xstd, xmean + xstd, color='red', alpha=0.1)
g1.ax_marg_y.axhline(ymean, color='red', ls=':')
g1.ax_marg_y.axhspan(ymean - ystd, ymean + ystd, color='red', alpha=0.1)
plt.tight_layout()
plt.savefig("GBRNoDFT.png",dpi=1200)
plt.clf()
print('GBR test predcition plot generated.')

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
plt.plot(x_z, lowess,color='r',linestyle=":")
plt.fill_between(x_z,ll,ul,alpha=.33)
plt.xlabel("Test $T_c$ in Kelvin")
plt.rcParams.update({'axes.labelsize':15})
plt.ylabel("$\\displaystyle\\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$",fontsize=15)
plt.tight_layout()

plt.savefig("ResidGBRNoDFT.png",dpi=1200)
plt.clf()
print('GBR residues plot generated.')

#ResidualDistPlot
sns.set_theme(style="whitegrid")
pred=GBR.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.histplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"))
plt.xlabel("$T_c^{\\mathrm{Pred}}-T_c$")
plt.xlabel("Test $T_c$ in Kelvin")
plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()

plt.savefig("ResidDistNoDFTGBR.png",dpi=1200)
plt.clf()
print('GBR residues dist plot generated.')

#ResidualKDEDistPlot
sns.set_theme(style="whitegrid")
pred=GBR.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.kdeplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"),fill=True)
plt.xlabel("$\\displaystyle \\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$")
plt.xlim((-3,3))
plt.ylabel("Distribution Density")

plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()

plt.savefig("ResidKDEDistGBRNODFT.png",dpi=1200)
plt.clf()
print('GBR residues KDE dist plot generated.')

#ResidualKDEDistPlot
sns.set_theme(style="whitegrid")
pred=GBR.predict(testData)
x_z, y_z = zip(*sorted(zip(testTc, (pred-testTc))))
sns.kdeplot( x=(pred-testTc)/testTc,palette=sns.color_palette("Blues"),fill=True)
plt.xlabel("$\\displaystyle \\frac{T_c^{\\mathrm{Pred}}-T_c}{T_c}$")
plt.ylabel("Distribution Density")
plt.rcParams.update({'axes.labelsize':15})
plt.tight_layout()
plt.savefig("ResidKDEDistGBRNODFTNoLim.png",dpi=1200)
plt.clf()
print('GBR residues KDE dist plot generated.')

##Feature Importance using SHAP
#Clean Descr Array
for i in range (0,len(descrNoDFT)):
    descrNoDFT[i]=descrNoDFT[i].replace('_',' ')
    descrNoDFT[i]=descrNoDFT[i].replace('Density Param for atom # 27','Cobalt Density')
    descrNoDFT[i]=descrNoDFT[i].replace('Density Param for atom # 28','Nickel Density')
    descrNoDFT[i]=descrNoDFT[i].replace('elenegTOTAL','$\\chi^{\\mathrm{Tot}}$')
    descrNoDFT[i]=descrNoDFT[i].replace('covrad of Atom 2','$r^{\\mathrm{Cov}}_{2}$')
    descrNoDFT[i]=descrNoDFT[i].replace('Atomic Numbers of Atom 1','$Z_{1}$')
    descrNoDFT[i]=descrNoDFT[i].replace('eleaffinTOTAL','$E^{\\mathrm{ea}}_{\\mathrm{Tot}}$')
    descrNoDFT[i]=descrNoDFT[i].replace('Density Param for atom # 26','Iron Density')
    descrNoDFT[i]=descrNoDFT[i].replace('Density Param for atom # 25','Manganese Density')
    descrNoDFT[i]=descrNoDFT[i].replace('# of Valence electrons of atom 2','$e^{\\mathrm{val}}$ of Atom 2')
    descrNoDFT[i]=descrNoDFT[i].replace('# of Valence electrons of Atom 2','$e^{\\mathrm{val}}$ of Atom 2')
    descrNoDFT[i]=descrNoDFT[i].replace('# of Valence electrons of atom 3','$e^{\\mathrm{val}}$ of Atom 3 ')
    descrNoDFT[i]=descrNoDFT[i].replace('# of Valence electrons of Atom 3','$e^{\\mathrm{val}}$ of Atom 3 ')
    descrNoDFT[i]=descrNoDFT[i].replace('TOTAL # of Valence electrons','Total $e^{\\mathrm{val}}$')
    descrNoDFT[i]=descrNoDFT[i].replace('#','\#')
    descrNoDFT[i]=descrNoDFT[i].replace('atmrad of atom 2','$r^{\\mathrm{Atom}}_2$')
    descrNoDFT[i]=descrNoDFT[i].replace('atmrad of Atom 2','$r^{\\mathrm{Atom}}_2$')
    descrNoDFT[i]=descrNoDFT[i].replace('atom','Atom')
    descrNoDFT[i]=descrNoDFT[i].replace('vdwrad of Atom 1','$r^{\\mathrm{vdw}}_1$')
    descrNoDFT[i]=descrNoDFT[i].replace('eleneg of Atom 2','$\chi^{(2)}$')
    descrNoDFT[i]=descrNoDFT[i].replace('eleaffin of Atom 1','$E^{\\mathrm{ea}}_1$')

#SHAP Beeswarm Plot (Mean SHAP Values, SHAP Values and SHAP Values for best 9 Descr.)
X2=pd.DataFrame(trainData,columns=descrNoDFT)
explainer=shap.TreeExplainer(GBR,X2)
plt.clf()
shap_values2=explainer(X2,check_additivity=False)

shap.summary_plot(shap_values2,show=False,plot_type="layered_violin",plot_size=(10,5))
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('TotRedDescrGBSHAP.png',dpi=600)
plt.clf()

shap.summary_plot(shap_values2,show=False,max_display=9,plot_type="layered_violin",plot_size=(10,5))
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.savefig('RedRedDescrGBSHAP.png',dpi=600)
plt.clf()
print('Feature importance plot for non-DFT Features generated.')

#Mabs + Den Plots
for k in ['Ferro Density','Cobalt Density', 'Nickel Density', 'Manganese Density',"Iron Density"]:
    if k == 'Ferro Density': 
        sns.set_theme(style="ticks",palette="Blues")
        hk=sns.displot(x=data[:,np.where(descrNoDFT==k)[0][0]],y=tc,palette="Blues",cbar=True,bins=(5,10),binrange=((-0.125,1.125),None))
        plt.xticks([-0.25,0.0,0.25,0.5,0.75,1])
        hk.set_axis_labels('Fraction of ferromagnetic Atoms','$\\displaystyle T_c$ in Kelvin')
    else: 
        sns.set_theme(style="ticks",palette="Blues")
        hk=sns.displot(x=data[:,np.where(descrNoDFT==k)[0][0]],y=tc,palette="Blues",cbar=True,bins=(4,10),binrange=((-0.125,0.875),None))
        hk.set_axis_labels(k.replace('Density','Fraction'),'$\\displaystyle T_c$ in Kelvin')
        plt.xticks([0.0, 0.25,0.5,0.75,1])
    
    plt.tight_layout()
    plt.xlim(np.min(data[:,np.where(descrNoDFT==k)[0][0]])-0.25,np.max(data[:,np.where(descrNoDFT==k)[0][0]])+0.25)
    plt.savefig('Tcvs'+k+'.png',dpi=1200)
    plt.clf()
    print('T_c vs '+k+' plot generated.')
    

sns.set_theme(style="ticks",palette="Blues")
hk=sns.displot(x=data[:,np.where(descrNoDFT=="Iron Density")[0][0]]+data[:,np.where(descrNoDFT=="Cobalt Density")[0][0]]+data[:,np.where(descrNoDFT=="Nickel Density")[0][0]],y=tc,palette="Blues",cbar=True,bins=(5,10),binrange=((-0.125,1.125),None))
plt.xticks([-0.25,0.0,0.25,0.5,0.75,1])
hk.set_axis_labels('Fraction of ferromagnetic Atoms','$\\displaystyle T_c$ in Kelvin')
plt.tight_layout()
plt.xlim(-0.25,1)
plt.savefig('Tcvs'+"Ferro2"+'.png',dpi=1200)
plt.clf()

    
#T_c vs Symmetry Code  
sns.set_theme(style='whitegrid')
h=sns.displot(x=data[:,np.where(descrNoDFT=='Symmetry Code')[0][0]],y=tc,cbar=True,bins=(int(np.max(data[:,np.where(descrNoDFT=='Symmetry Code')[0][0]])+1),10),binrange=((np.min(data[:,np.where(descrNoDFT=='Symmetry Code')[0][0]])-0.5,np.max(data[:,np.where(descrNoDFT=='Symmetry Code')[0][0]])+0.5),None))
plt.xlabel('Symmetry Code')
plt.ylabel('$\\displaystyle T_c$ in Kelvin')
plt.tight_layout()
plt.savefig('TcvsSymCode.png',dpi=1200)
plt.clf()
print('T_c vs Symmetry Code plot generated.')
    
#LassoLars
best=linear_model.LassoLarsCV( max_iter=100000,  cv=4, max_n_alphas=100000, eps=1e-16, copy_X=True)
best.fit(trainData,trainTc)
modelEvalReg(best,'LassoLarsReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

#Lasso
best=linear_model.LassoCV(eps=0.00001, n_alphas=1000, precompute='auto', max_iter=10000, tol=0.00001,cv=4)
LL=best#Save for ind. Class later
modelEvalReg(best,'LassoReg no DFT data',testTc,trainTc,testData,trainData,data,tc)

#LinReg
best=LinearRegression()
modelEvalReg(best,'LinReg NoDFT',testTc,trainTc,testData,trainData,data,tc)

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



#xgboost
xgBoost=XGBRegressor()
xgBoostSet={'n_estimators': [4,10,100,1000,5000],'eta':[0.01,0.05,0.1,0.2,0.3],"lambda":[1.1,1.05,1.0],"eval_metric":["mse"]}
best=opt(XGBRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'XGBost',testTc,trainTc,testData,trainData,data,tc)

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

#KNN
params={'n_neighbors':[2,3,4,5,6,10,15,20,25,30,40,50],'p':[1,2,3,4],'leaf_size': [5,10,20,30,40,50,60,70]}
best=opt(KNNClass(),'f1',trainData,trainTc,params)
modelEvalClass(best,'KNNClass NoDFT',testTcClass,trainTcClass,testData,trainData,data,TcClass)


#DTC
params={ 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'DescisionTreesClass no DFT data',testTcClass.astype(int),trainTcClass.astype(int),testData,trainData,data,TcClass.astype(int))

#LogReg
params={'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.1,0.5,1],'solver': ['liblinear','saga']}
best=opt(LogisticRegression(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'Logistic no DFT data',testTcClass.astype(int),trainTcClass.astype(int),testData,trainData,data,TcClass.astype(int))

#XGBClass
params={'n_estimators': [4,10,100,1000,5000],'eta':[0.01,0.05,0.1,0.2,0.3],"lambda":[1.1,1.05,1.0]}
best=opt(XGBC(),'f1',trainData,trainTcClass,params)
xgbMod=best
modelEvalClass(best,'XGBClass no DFT data',testTcClass,trainTcClass,testData,trainData,data,TcClass)


#GBClass
params={'n_estimators': [4,10,100,1000,5000],'learning_rate':[0.01,0.05,0.1,0.15,0.2]}
best=opt(GradientBoostingClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'GBClass NoDFt',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#RF
params={'n_estimators': [10000],'criterion': ['gini','logloss']}
best=opt(RandomForestClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'RandForClass No DFT',testTcClass,trainTcClass,testData,trainData,data,TcClass)


#Ind Lasso
modelEvalIndClass(LL,'IndirectLASSOClass no DFT data',testTc,trainTc,testData,trainData,data,tc)

#Ind ETR
modelEvalIndClass(ETR,'IndirectETR Class no DFT data',testTc,trainTc,testData,trainData,data,tc)

##Confusion Matrices
#Indirect via ETR
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(ETR.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.tight_layout()
plt.savefig('ConfIndETRNoDFT.png')
plt.clf()

#XGBOost
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),xgbMod.predict(testData).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.tight_layout()
plt.savefig('ConfIndXGBNoDFT.png')
plt.clf()

#Indirect via LASSO
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(LL.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.tight_layout()
plt.savefig('ConfIndLLNoDFT.png')

#Direct via ETC
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),ETC.predict(testData).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low $T_c$','High $T_c$'])
cmd.plot()
plt.grid()
plt.tight_layout()
plt.savefig('ConfDirETCNoDFT.png')
