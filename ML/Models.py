#This script uses processed JuHemd data to train ML-Models to make qualitative and quantitative predictions of a test Tc set.
#Different ML models are compared using a GridSearch algorithm, CV scores, test and train scores and - in the classification case - accuracies.
#Mutliple figures are generated for the purpose of displaying model performance and properties of the data.
#- Robin Hilgers, 2022
#Published on Zenodo
#--------------------------------------------------------------------------------
# Copyright (c) 2022 Peter Grünberg Institut, Forschungszentrum Jülich, Germany
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
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor as KNN
#Data Treatment/Splitting
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

#Enter path for data to be loaded
path=input('Enter valid path to files with Heuslers Data (Descriptors.txt ,Tc.txt, Data.txt and DataClearedFromDTF.txt containing folder): ')

#Load Data
try:
    tc = np.genfromtxt(path+'/Tc.txt')
    data = np.genfromtxt(path+'/Data.txt')[:, 1:]
    descr=np.genfromtxt(path+'/Descriptors.txt',dtype=str,delimiter=',',comments='$')[1:]
    dataNoDFT=np.genfromtxt(path+'/DataClearedFromDTF.txt')[:, 1:]
    print('Path Found')
except:
    tc=np.genfromtxt("/home/hilgers/PhD/Heuslers/2022_Heusler_Paper_Prep/Data Publication/ML-Ready KKR JuHemd Database/GGA/Tc.txt")
    data=np.genfromtxt("/home/hilgers/PhD/Heuslers/2022_Heusler_Paper_Prep/Data Publication/ML-Ready KKR JuHemd Database/GGA/Data.txt")[:,1:]
    dataNoDFT=np.genfromtxt('/home/hilgers/PhD/Heuslers/2022_Heusler_Paper_Prep/Data Publication/ML-Ready KKR JuHemd Database/GGA/DataClearedFromDTF.txt')[:, 1:]
    descr=np.genfromtxt('/home/hilgers/PhD/Heuslers/2022_Heusler_Paper_Prep/Data Publication/ML-Ready KKR JuHemd Database/GGA/Descriptors.txt',comments='$',dtype=str,delimiter=',')[1:]
    print('Path not found, fallback solution.')

#Split and shuffle data
trainData,testData,trainTc,testTc=train_test_split(data,tc,test_size=0.2,shuffle=True ,random_state=3141592)#31415

#Small functions for easy evaluation of regression models
def modelEvalReg(model,nameModel,ytest,ytrain,xtest,xtrain,data,tc):
    model.fit(xtrain,ytrain)
    cv=cross_val_score(model,trainData,trainTc,cv=5,scoring='r2')
    testScore=r2_score(ytest,model.predict(xtest))
    trainScore=r2_score(ytrain,model.predict(xtrain))
    print('Model performance for '+ nameModel)
    print('CV: ')
    print(np.mean(cv))
    print('CV: ')
    print(cv)
    print('R2-Test: ')
    print(testScore)
    print('R2-Train:')
    print(trainScore)
    return

#Evaluation function for classification models
def modelEvalClass(model,nameModel,ytest,ytrain,xtest,xtrain,data,tc):
    model.fit(xtrain,ytrain)
    cv=cross_val_score(model,xtrain,ytrain.astype(int),cv=5,scoring='f1')
    testScore=f1_score(ytest.astype(int),model.predict(xtest).astype(int))
    trainScore=f1_score(ytrain.astype(int),model.predict(xtrain).astype(int))
    print('Model performance for '+ nameModel)
    print('CV: ')
    print(np.mean(cv))
    print(cv)
    print('F1-Test: ')
    print(testScore)
    print('F1-Train:')
    print(trainScore)
    print('Test Acc.:')
    print(accuracy_score(ytest, model.predict(xtest)))
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
    print('Model performance for '+ nameModel)
    print('F1-Test: ')
    print(testScore)
    print('F1-Train:')
    print(trainScore)
    print('Test Acc.:')
    print(accuracy_score(classify(ytest,thres,labels).astype(int), classify(model.predict(xtest),thres,labels).astype(int)))
    return

#Function to automate optimization and selection
def opt(model, scor,data,tc,params):
    search = GridSearchCV(model, params, cv=5, scoring=scor)
    search.fit(data, tc)
    return search.best_estimator_

##With DFT Data
#Extra Trees
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(ExtraTreesRegressor(),'r2',trainData,trainTc,params)
ETR=best#Save for ind. Class later
modelEvalReg(best,'ExtraTrees',testTc,trainTc,testData,trainData,data,tc)

#LassoLars
best=linear_model.LassoLarsCV(max_iter=100000,  cv=5, max_n_alphas=100000, eps=1e-16, copy_X=True)
LL=best#Save for ind. Class later
LL.fit(trainData,trainTc)
modelEvalReg(best,'LassoLars',testTc,trainTc,testData,trainData,data,tc)

#Lasso
best=linear_model.LassoCV(eps=0.00001, n_alphas=1000, precompute='auto', max_iter=10000, tol=0.00001,cv=5)
reg=best#Save for Feature Importance
modelEvalReg(best,'Lasso',testTc,trainTc,testData,trainData,data,tc)

#LinReg
best=LinearRegression()
modelEvalReg(best,'Lin Reg.',testTc,trainTc,testData,trainData,data,tc)

#DTR
params={'criterion' : ['squared_error'], 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'Decision Tree',testTc,trainTc,testData,trainData,data,tc)

#KNN
params={'n_neighbors':[2,3,4,5,6,10,15],'p':[1,2,3,4],'leaf_size': [5,10,20,30,40,50,60,70]}
best=opt(KNN(),'r2',trainData,trainTc,params)
modelEvalReg(best,'KNN',testTc,trainTc,testData,trainData,data,tc)

#RTR
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(RandomForestRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'RTR',testTc,trainTc,testData,trainData,data,tc)

##Move To classification
#Prerequisites
labels=['0','1'] # Low Tc and High Tc Classific.
thres=[273.15+200]
testTcClass= classify(testTc,thres,labels).astype(int)
trainTcClass= classify(trainTc,thres,labels).astype(int)
TcClass= classify(tc,thres,labels).astype(int)

#ETC
params={'n_estimators': [10000],'criterion': ['gini']}
best=opt(ExtraTreesClassifier(),'f1',trainData,trainTcClass,params)
ETC=best
modelEvalClass(best,'ETC',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#DTC
params={ 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'DTC',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#LogReg
params={'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.1,0.5,1],'solver': ['liblinear','saga']}
best=opt(LogisticRegression(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'Logistic',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#Ind LassoLars
modelEvalIndClass(LL,'Indirect LASSOLars Class',testTc,trainTc,testData,trainData,data,tc)

#Ind ETR
modelEvalIndClass(ETR,'Indirect ETR Class',testTc,trainTc,testData,trainData,data,tc)

##Figure generation
#testTc ETR
pred=ETR.predict(testData)
f, ax = plt.subplots(figsize=(5, 5))
df = pd.DataFrame({"Predicted $T_c$ in Kelvin":pred})
df["Test $T_c$ in Kelvin"]=pd.Series(testTc)
sns.set_theme(style="whitegrid")
g1=sns.jointplot(data=df,x="Test $T_c$ in Kelvin",y="Predicted $T_c$ in Kelvin",kind='reg',dropna = True)
g1.ax_joint.plot([0,np.max(testTc)], [0,np.max(testTc)], 'r-', linewidth = 2)
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
plt.ylabel("$\\frac{T_c^{Pred}-T_c}{T_c}$")
plt.savefig("ResidETR.png",dpi=1200)
plt.clf()
print('ETR residues plot generated.')

#Test Tc LASSOLars
pred=LL.predict(testData)
f, ax = plt.subplots(figsize=(5, 5))
df = pd.DataFrame({"Predicted $T_c$ in Kelvin":pred})
df["Test $T_c$ in Kelvin"]=pd.Series(testTc)
sns.set_theme(style="whitegrid")
g1=sns.jointplot(data=df,x="Test $T_c$ in Kelvin",y="Predicted $T_c$ in Kelvin",kind='reg',dropna = True)
g1.ax_joint.plot([0,np.max(testTc)], [0,np.max(testTc)], 'r-', linewidth = 2)
plt.savefig("LL.png",dpi=1200)
plt.clf()
print('LASSOLars test plot generated.')

#Histogram of Atomic #
dataOccup=data[:,59:63]
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
    bins=int(np.max(data[:,59:63])),
)
ax.set(xlabel='Atomic Number')
plt.grid()
sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
plt.savefig("Hist.png",dpi=1200)
plt.clf()
print('Histogram generated.')

#Feature Importance using LASSO
X=pd.DataFrame(data)
y=pd.DataFrame(tc)
reg.fit(data, tc)
print("Score using 5-Fold CV on WHOLE data set: %f" %np.mean(cross_val_score(reg,data,tc,cv=5,scoring='r2')))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated " +  str(sum(coef == 0)) + " variables")
ax1=plt.subplot()
reg_coef, descr = zip(*sorted(zip(coef, descr)))
descr=np.array(descr)
for i in range (0,len(descr)):
    descr[i]=descr[i].replace('#','\#')
reg_coef=pd.Series(reg_coef,index = X.columns)
mpl.rcParams['figure.figsize'] = (8.0, 10.0)
#exist=np.where(np.abs(reg_coef)>0.01)[0]
exist=np.zeros(0,dtype=int)
for i in range (0,len(reg_coef)):
    if np.abs(reg_coef.iloc[i])>0.0001:
        exist=np.append(exist,int(i))
print('Feature Coeffs:')
print(reg_coef[exist])
reg_coef.iloc[exist].plot(kind = "barh")
ax1.set_yticklabels(descr[exist])
plt.grid()
plt.xlabel('Coefficient Weight')
plt.tight_layout()
plt.savefig("FeatureImportance.png",dpi=600)
plt.clf()
print('Feature importance plot generated.')

##Confusion Matrices
#Indirect via ETR
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(ETR.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low Tc','High Tc'])
cmd.plot()
plt.savefig('ConfIndETR.png')
plt.clf()

#Indirect via LASSOLars
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(LL.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low Tc','High Tc'])
cmd.plot()
plt.savefig('ConfIndLL.png')

#Direct via ETC
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),ETC.predict(testData).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low Tc','High Tc'])
cmd.plot()
plt.savefig('ConfDirETC.png')

##Without DFT
#Shuffle and split data
trainData,testData,trainTc,testTc=train_test_split(dataNoDFT,tc,test_size=0.2,shuffle=True ,random_state=31415)
#Extra Trees
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(ExtraTreesRegressor(),'r2',trainData,trainTc,params)
ETR=best#Save for ind. Class later
modelEvalReg(best,'ExtraTrees no DFT data',testTc,trainTc,testData,trainData,data,tc)

#LassoLars
best=linear_model.LassoLarsCV( max_iter=100000,  cv=5, max_n_alphas=100000, eps=1e-16, copy_X=True)
LL=best#Save for ind. Class later
LL.fit(trainData,trainTc)
modelEvalReg(best,'LassoLars no DFT data',testTc,trainTc,testData,trainData,data,tc)

#Lasso
best=linear_model.LassoCV(eps=0.00001, n_alphas=10000, precompute='auto', max_iter=1000000, tol=0.00001,cv=5)
modelEvalReg(best,'Lasso no DFT data',testTc,trainTc,testData,trainData,data,tc)

#LinReg
best=LinearRegression()
modelEvalReg(best,'Lin Reg.',testTc,trainTc,testData,trainData,data,tc)

#DTR
params={'criterion' : ['squared_error'], 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'Decision Tree no DFT data',testTc,trainTc,testData,trainData,data,tc)

#KNN
params={'n_neighbors':[2,3,4,5,6,10,15],'p':[1,2,3,4],'leaf_size': [5,10,20,30,40,50,60,70]}
best=opt(KNN(),'r2',trainData,trainTc,params)
modelEvalReg(best,'KNN no DFT data',testTc,trainTc,testData,trainData,data,tc)

#RTR
params={'n_estimators': [10,100,1000,10000],'criterion' : ['squared_error']}
best=opt(RandomForestRegressor(),'r2',trainData,trainTc,params)
modelEvalReg(best,'RTR no DFT data',testTc,trainTc,testData,trainData,data,tc)

##Move To classification
#Prerequisites
labels=['0','1'] # Low Tc and High Tc Classific.
thres=[273.15+200]
testTcClass= classify(testTc,thres,labels).astype(int)
trainTcClass= classify(trainTc,thres,labels).astype(int)
TcClass= classify(tc,thres,labels).astype(int)

#ETC
params={'n_estimators': [10000],'criterion': ['gini']}
best=opt(ExtraTreesClassifier(),'f1',trainData,trainTcClass,params)
ETC=best# For confu Matrix later
modelEvalClass(best,'ETC no DFT data',testTcClass,trainTcClass,testData,trainData,data,TcClass)

#DTC
params={ 'max_features':['sqrt','log2','auto']}
best=opt(DecisionTreeClassifier(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'DTC no DFT data',testTcClass.astype(int),trainTcClass.astype(int),testData,trainData,data,TcClass.astype(int))

#LogReg
params={'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.1,0.5,1],'solver': ['liblinear','saga']}
best=opt(LogisticRegression(),'f1',trainData,trainTcClass,params)
modelEvalClass(best,'Logistic no DFT data',testTcClass.astype(int),trainTcClass.astype(int),testData,trainData,data,TcClass.astype(int))

#Ind LassoLars
modelEvalIndClass(LL,'Indirect LASSOLars Class no DFT data',testTc,trainTc,testData,trainData,data,tc)

#Ind ETR
modelEvalIndClass(ETR,'Indirect ETR Class no DFT data',testTc,trainTc,testData,trainData,data,tc)


##Confusion Matrices
#Indirect via ETR
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(ETR.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low Tc','High Tc'])
cmd.plot()
plt.savefig('ConfIndETRNoDFT.png')
plt.clf()

#Indirect via LASSOLars
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),classify(LL.predict(testData),thres,labels).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low Tc','High Tc'])
cmd.plot()
plt.savefig('ConfIndLLoDFT.png')

#Direct via ETC
cm=confusion_matrix(classify(testTc,thres,labels).astype(str),ETC.predict(testData).astype(str),labels=['0','1'])
cmd = ConfusionMatrixDisplay(cm,display_labels=['Low Tc','High Tc'])
cmd.plot()
plt.savefig('ConfDirETCoDFT.png')