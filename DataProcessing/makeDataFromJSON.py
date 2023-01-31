#This scirpt evaluates 'JuHemd.json', processes, cleans and randomizes the data and adds additional descriptors in a ML-ready format.
#- Robin Hilgers, 2022
#Published on Zenodo
#--------------------------------------------------------------------------------
# Copyright (c) 2023 Peter Grünberg Institut, Forschungszentrum Jülich, Germany
# This file is available as free software under the conditions
# of the MIT license as expressed in the LICENSE file in more detail.
#--------------------------------------------------------------------------------



#Imports
import numpy as np
import mendeleev as me
import csv
import json
import sklearn as sk
#Decide whether GGA or LDA data should be used
path=input('Enter GGA if you want to extract GGA data from the JuHemd.json file or enter LDA if you want to extract LDA data from the JuHemd.json file:    ')
if path=='GGA':
    LoG='DSIMG-TcMCb'
elif path=='LDA':
    LoG='DSIML-TcMCb'
else: print('Invalid input. This will result in an error.')


#Function to apend newly occured errors to the total error array. 
#Also ensures every compound which occured as an error only occurs once. 
#Error Array contains label of failed compound.
def errHandl(oldErr,newErr):
    for i in range (0,len(newErr)):
        if newErr.size>0:
            if not np.any(newErr[i]==oldErr):
                oldErr=np.append(oldErr,newErr[i])
    uptArr=np.unique(oldErr)
    return uptArr

#Retuns the amount of entries/disorders for one compound with identical elmental label.
def jReturn(i,data):
    try:
        return len(data[i][LoG][:])
    except:
        return 1

#Returns size of the dataset.
def dataSize(data):
    size=0
    keys=data.keys()
    for i in keys:
        size=size+jReturn(i,data)
    return int(size)

#Loads/Pre-Processes atomistic data (arrayAtoms) as well as the loaded destciptor names (qstrings). 
#ptable.csv is provided with the corresponding data publication. 
def initAtomData():
    with open('ptable.csv') as csvfile:
        atomicData = csv.reader(csvfile, delimiter=',')
        arrayAtom = np.zeros(0)
        for row in atomicData:
            arrayAtom = np.append(arrayAtom, row)
    arrayAtom = np.reshape(arrayAtom, (-1, 21))
    qStrings = np.zeros(0, dtype=str)
    qStrings = arrayAtom[0, :]
    arrayAtom = np.delete(arrayAtom, (0), axis=0)
    arrayAtom = np.where(arrayAtom == 's', '1', arrayAtom)
    arrayAtom = np.where(arrayAtom == 'p', '2', arrayAtom)
    arrayAtom = np.where(arrayAtom == 'd', '3', arrayAtom)
    arrayAtom = np.where(arrayAtom == 'f', '4', arrayAtom)
    return arrayAtom, qStrings

#Returns values of compound i in it's jth occurence in the dataset. Which value from the data set is returned is determined by a keyword "elementName".
#If the value couldn't be retrieved the compounds name is returned for error handling. 
def returnValues(elementName,dataName,j,i):
    newElement = np.zeros(0)
    errorCompounds = np.zeros(0)
    if elementName=='systemSites-S':
        try:
            extr = dataName[i][LoG][j]['system_sites']
            newElement = np.append(newElement, extr)
            [sites, occup, magnet, labels] = [newElement[0::4], newElement[1::4], newElement[2::4], newElement[3::4]]
        except:
            if j == 0: errorCompounds = np.append(errorCompounds, i)
            if j != 0: errorCompounds = np.append(errorCompounds  + str(j), i)
            sites = np.append(sites, '')
        return sites ,errorCompounds
    if elementName=='systemSites-O':
        try:
            extr = dataName[i][LoG][j]['system_sites']
            newElement = np.append(newElement, extr)
            [sites, occup, magnet, labels] = [newElement[0::4], newElement[1::4], newElement[2::4], newElement[3::4]]
        except:
            if j == 0: errorCompounds = np.append(errorCompounds, i)
            if j != 0: errorCompounds = np.append(errorCompounds  + str(j), i)
            occup = np.append(occup, '')
        return occup,errorCompounds
    if elementName=='systemSites-M':
        try:
            extr = dataName[i][LoG][j]['system_sites']
            newElement = np.append(newElement, extr)
            [sites, occup, magnet, labels] = [newElement[0::4], newElement[1::4], newElement[2::4], newElement[3::4]]
        except:
            if j == 0: errorCompounds = np.append(errorCompounds, i)
            if j != 0: errorCompounds = np.append(errorCompounds + str(j), i)
            magnet = np.append(magnet, '')
        return magnet,errorCompounds
    if elementName=='systemSites-L':
        try:
            extr = dataName[i][LoG][j]['system_sites']
            newElement = np.append(newElement, extr)
            [sites, occup, magnet, labels] = [newElement[0::4], newElement[1::4], newElement[2::4], newElement[3::4]]
        except:
            if j == 0: errorCompounds = np.append(errorCompounds, i)
            if j != 0: errorCompounds = np.append(errorCompounds  + str(j), i)
            labels = np.append(labels, '')
        return labels,errorCompounds
    if elementName == 'lattice_constant':
        try:
            newElement = np.append(newElement, dataName[i][LoG][j]['str']['lattice_constant'])
        except:
            if j == 0: errorCompounds = np.append(errorCompounds, i)
            if j != 0: errorCompounds = np.append(errorCompounds + str(j), i)
            newElement = np.append(newElement, '')
        return newElement, errorCompounds
    if elementName == 'label':
        try:
            temp = dataName[i][LoG][j]['str']['label']
            newElement = np.append(newElement, temp)
        except:
            if j == 0: errorCompounds = np.append(errorCompounds, i)
            if j != 0 : errorCompounds = np.append(errorCompounds + str(j), i)
            newElement = np.append(newElement, '')
        return newElement, errorCompounds
    if elementName == 'etotal':
        try:
            newElement = np.append(newElement, dataName[i][LoG][j]['system_dft']['results']['quantities']['etotal (Ry)'])
        except:
            if j==0: errorCompounds = np.append(errorCompounds, i)
            if j!=0: errorCompounds = np.append(errorCompounds+str(j), i)
            newElement = np.append(newElement, 0)
        return newElement, errorCompounds
    if elementName == 'formula' and j==0:
        newElement = np.append(newElement, str(i).replace('_', ''))
        return newElement, errorCompounds
    if elementName == 'formula' and j!=0:
        newElement = np.append(newElement, str(i).replace('_', '')+str(j))
        return newElement, errorCompounds
    if elementName == 'magnetic_state':
        try:
            temp = dataName[i][LoG][j]['system_magnetic-structure-factors'][:]
        except:
            temp = ''
            if j == 0: errorCompounds = np.append(errorCompounds, i)

            if j != 0: errorCompounds = np.append(str(errorCompounds) + str(j), i)
        temp2 = ''
        for k in range(1, len(temp[:])):
            temp2 = temp2 + temp[k][1]
        newElement = np.append(newElement, temp2)
        return newElement, errorCompounds
    if elementName == 'Tc':  
        try:
            k = dataName[i][LoG][j]['data']['resval']
            if len(k) > 1:
                k = k[0]
            newElement = np.append(newElement, k)
        except:
            if j == 0: errorCompounds = np.append(errorCompounds, i)
            if j != 0: errorCompounds = np.append(errorCompounds + ' ' + str(j), i)
            newElement = np.append(newElement, 0)
        return newElement, errorCompounds

#Loads a single descriptor for all compounds and returns the value as well as the compounds the value couldn't be loaded for.
def load_single_element(elementName, dataName):
    newRes=np.zeros(0)
    err=np.zeros(0)
    keys = dataName.keys()
    for i in keys:
        new,errN=returnValues(elementName, dataName, 0, i)
        newRes=np.append(newRes,new)
        err=np.append(err,errN)
    nArr=np.zeros(0,dtype=int)
    for i in keys:
        nArr=np.append(nArr,int(jReturn(i,dataName)))
    k=0
    for i in keys:
        for j in range (1,nArr[k]):
            new, errN = returnValues(elementName, dataName, j, i)
            newRes = np.append(newRes, new)
            err = np.append(err, errN )
        k = k + 1
    return newRes,err

#Returns the # of Valence electrons as stored in the mendeleev database. 
def valenceEl(Zx):
    vale = np.zeros(len(Zx), dtype=int)
    for i in range(0, len(Zx)):
        if Zx[i] > 0.0:
            le = me.element(int(Zx[i]))
            vale[i] = le.nvalence(method=None)
    return vale

#Appends a new descriptor to the descriiptor array. Also adds name of the descriptor to the nameList.
def append_single_element(dataName, newElement, nameList, dataFrame):
    nameList = np.append(nameList, dataName)
    if np.size(dataFrame) == 0:
        dataFrame = newElement
    else:
        dataFrame = np.c_[dataFrame, newElement]
    return dataFrame, nameList

#Loads single descriptor form the database as well as appending it to the dataFrame and adding the descriptor name to the nameList. At this point
#also default entries are contained for compounds which failed to load. Every compound stored in the err array gets removed entirely from the descriptor array later. 
def load_and_append_single_element(eleName, dataName, nameList, dataFrame):
    new_ele, err = load_single_element(eleName, dataName) 
    dataFrame, nameList = append_single_element(eleName, new_ele, nameList, dataFrame)
    return dataFrame, nameList,err

#Splits given Heusler labels to elemental symbols. 
def splitStrings(labels):
    for i in range (0,len(labels)):
        labels[i]=labels[i].replace('1','')
        labels[i]=labels[i].replace('2','')
        labels[i]=labels[i].replace('3','')
        labels[i]=labels[i].replace('4','')
        labels[i]=labels[i].replace('5','')
        labels[i]=labels[i].replace('6','')
        labels[i]=labels[i].replace('7','')
        labels[i]=labels[i].replace('8','')
        labels[i]=labels[i].replace('9','')
        labels[i]=labels[i].replace('0','')
    str1=np.zeros(len(labels),dtype='|S5')
    str2=np.zeros(len(labels),dtype='|S5')
    str3=np.zeros(len(labels),dtype='|S5')
    str4=np.zeros(len(labels),dtype='|S5')
    boolArr=np.zeros(8,dtype=bool)
    for i in range(0, len(labels)):
        for j in range (0,len(labels[i])-1):
            boolArr[j]=labels[i][j].isupper()
        arr=np.where(boolArr==True)
        str1[i]=labels[i][arr[0][0:4][0]: arr[0][0:4][1]]
        str2[i]=labels[i][arr[0][0:4][1]: arr[0][0:4][2]]
        str3[i]=labels[i][arr[0][0:4][2]: arr[0][0:4][3]]
        str4[i]=labels[i][arr[0][0:4][3]: arr[0][0:4][3]+2]
    return str1, str2, str3, str4

#Handles vacancies (Xx) as well as reformatting single character elemental symbols before getting the atomic # from the mendeleev database. Vacancies have atomic # 0.  
def handNameStringHandler(someStrng):
    if  str(someStrng).find("'")!=-1:
        redStr=str(someStrng)[2:4]
    else: 
        redStr=str(someStrng)
    if (not redStr == ''):
        if (redStr=='Xx'):
            return 0
        elif redStr=="V'" or redStr=='V':
            return 23
        elif redStr=="W'" or redStr=='W' :
            return 74
        elif redStr=="Y'" or redStr=='Y':
            return 39
        elif redStr=="B'" or redStr=='B':
            return 5
        elif redStr=="N'" or redStr=='N':
            return 7
        else:
            return me.element(redStr).atomic_number
    else:
        return 0



#Split strings for strings having a different type
def splitStringsExtra(lables):
    for i in range(len(lables)):
        lables[i]=lables[i].replace('1','')
        lables[i]=lables[i].replace('2','')
        lables[i]=lables[i].replace('3','')
        lables[i]=lables[i].replace('4','')
        lables[i]=lables[i].replace('5','')
        lables[i]=lables[i].replace('6','')
        lables[i]=lables[i].replace('7','')
        lables[i]=lables[i].replace('8','')
        lables[i]=lables[i].replace('9','')
        lables[i]=lables[i].replace('0','')
    str1=np.zeros(len(lables),dtype='U100')
    str2=np.zeros(len(lables),dtype='U100')
    str3=np.zeros(len(lables),dtype='U100')
    str4=np.zeros(len(lables),dtype='U100')
    for i in range (0,len(lables)):
            boolArr=np.zeros(8,dtype=bool)
            for j in range (0,len(lables[i])):
                boolArr[j]=lables[i][j].isupper()
            num=boolArr.sum()
            arr=np.zeros(0,dtype=int)
            for l in range (0,len(boolArr)):
                if boolArr[l]==True: arr=np.append(arr,l)
            for k in range(0,num):
                if k==0: 
                    if num-1==k:str1[i]=lables[i][arr[0]:]
                    else: str1[i]=lables[i][arr[0]:arr[1]]
                if k==1: 
                    if num-1==k: str2[i]=lables[i][arr[1]:]
                    else: str2[i]=lables[i][arr[1]:arr[2]]
                if k==2: 
                    if num-1==k: str3[i]=lables[i][arr[2]:]
                    else: str3[i]=lables[i][arr[2]:arr[3]]
                if k==3: str4[i]=lables[i][arr[3]:]
            for k in range (4,num):
                if k==0: str1[i]=''
                if k==1: str2[i]=''
                if k==2: str3[i]=''
                if k==3: str4[i]=''
    return str1,str2,str3,str4
    
#Assignes and returns atomic numbers given the Heusler lables.
def giveAtomicNumber(labels):
    if labels.dtype=='<U100':
        A,B,C,D=splitStringsExtra(labels)
        A1=np.zeros(0,dtype=int)
        B1=np.zeros(0,dtype=int)
        C1=np.zeros(0,dtype=int)
        D1=np.zeros(0,dtype=int)
        for i in range (0,len(labels)):
            A1 = np.append(A1,handNameStringHandler(A[i]))
            B1 = np.append(B1,handNameStringHandler(B[i]))
            C1 = np.append(C1,handNameStringHandler(C[i]))
            D1 = np.append(D1,handNameStringHandler(D[i]))
        return A1, B1, C1, D1
    else:
        A, B, C, D = splitStrings(labels)
        A1=np.zeros(0)
        B1=np.zeros(0)
        C1=np.zeros(0)
        D1=np.zeros(0)
        for i in range (0,len(labels)):
            A1 = np.append(A1,handNameStringHandler(A[i]))
            B1 = np.append(B1,handNameStringHandler(B[i]))
            C1 = np.append(C1,handNameStringHandler(C[i]))
            D1 = np.append(D1,handNameStringHandler(D[i]))
    return A1, B1, C1, D1

#Checks and returns the # of unique elements in an Heusler alloy by the Heuslers compound label.
def isolateUniqueEle(labels):
    eList = np.zeros(0, dtype=str)
    for i in range(0, len(labels)):
        if not np.any(labels[i] == eList):
            eList = np.append(eList, labels[i])
    return eList

#Loads fractional appearance of the individual elements in the Heusler Alloy. Given the compound i and its occurence j. 
def load_fractions(data, i,j):
    S,err=returnValues('systemSites-S',data,j,i)
    O,errN=returnValues('systemSites-O',data,j,i)
    err = errHandl(err, errN)
    M,errN=returnValues('systemSites-M',data,j,i)
    err = errHandl(err, errN)
    L,errN=returnValues('systemSites-L',data,j,i)
    err = errHandl(err, errN)
    uniqe = isolateUniqueEle(L)
    numUnique = len(uniqe)
    fractionsArray = np.zeros(0)
    for i in range(0, numUnique):
        indicesArr = np.argwhere(L == uniqe[i])
        fractionsArray = np.append(fractionsArray, np.sum(O[indicesArr].astype(float)) / 4)
    return uniqe, fractionsArray,err

#Loads fractional occupations of the elements in all Heusler alloys from the data. 
def load_fractions_array(data):
    unique = np.zeros((dataSize(data), 4), dtype='U24')
    err=np.zeros(0)
    fractionsArray = np.zeros((dataSize(data), 4))
    k=0
    for i in data.keys():
        try:
            u, f,errN = load_fractions(data, i,0)
            err=np.append(err,errN)
        except:
            err = np.append(err, i)
            u=['']
            f=[0]
        integ = len(u)
        unique[k, 0:integ] = u
        fractionsArray[k, 0:integ] = f
        k=k+1
    for i in data.keys():
        u=jReturn(i,data)
        for j in range (1,u):
            try:
                u, f,errN = load_fractions(data, i,j)
                err = np.append(err, errN)
            except:
                err = np.append(err, i+str(j))
                u=['']
                f=[0]
            integ = len(u)
            unique[k, 0:integ] = u
            fractionsArray[k, 0:integ] = f
            k=k+1
    return unique, fractionsArray, err

#Returns unique sites 
def isoltateUniqueSites(sites):
    uniqueSites = np.zeros(0, dtype=int)
    for i in range(0, len(sites)):
        if not np.any(sites[i] == uniqueSites):
            uniqueSites = np.append(uniqueSites, sites[i])
    return uniqueSites


#Created a stochiometry which is a string of integers and quantifies the concentration of the unique elements in the Heusler. A 5th string is added for the vacuum. 
def load_stochiometry(data):
    strArr = np.zeros(0,dtype=str)
    sums = np.zeros(0)
    labels, frac,err = load_fractions_array(data)
    for i in range(0, len(frac[:, 0])):
        sums = np.append(sums, np.sum(frac[i, :]))
    vac = np.abs((sums -1) / 4).astype(int)
    intA = (frac.astype(float) * 10).astype(int)
    for i in range (0,len(intA[:,0])):
        strArr = np.append(strArr,(intA[i, 0]).astype(str) + (intA[i, 1]).astype(str) + (intA[i, 2]).astype(str) + (intA[i, 3]).astype(str) + (vac[i] * 10).astype(str))
    return strArr,err

#Gives 1 or 0 value if the given element is a rare-earth element.
def is_RE(el):
    el=str(el)
    if  el in ['Sc' ,'Y' , 'La' , 'Ce' ,'Pr' ,'Nd' , 'Pm' ,'Sm' ,'Eu' , 'Gd' ,'Tb' ,'Dy' ,'Ho','Er' ,'Tm', 'Yb', 'Lu']:
        return 1
    else:
        return 0

#Same as previou function but does it for an array of elements. 
def is_REArray(elementArr):
    intArray = np.zeros(0, dtype=int)
    for i in elementArr:
        intArray = np.append(intArray, is_RE(i))
    return intArray

#Loads the concentration of RE elements from the Heusler compounds formula from the dataset. 
def load_conc_rare_earth(formula, data):
    a, b, c, d = splitStrings(formula)
    A = is_REArray(a)
    B = is_REArray(b)
    C = is_REArray(c)
    D = is_REArray(d)
    sum = (A + B + C + D)/4
    return sum

#Checks if given Element is ferromagnetic.
def is_Ferro(ele):
    if  'Fe' in str(ele)  or  'Co' in str(ele) or 'Ni' in str(ele):
        return 1
    else:
        return 0

#Same as previous function but does it for an array of elements. 
def is_FerroArray(elementArray):
    intArray = np.zeros(0)
    for i in elementArray:
        intArray = np.append(intArray, is_Ferro(i))
    return intArray

#Gives the concentration of Ferrfomagnetic elements and loads it from the Heusler compounds formula from the dataset. 
def load_Ferro_conc(formula, data):
    a, b, c, d = splitStrings(formula)
    A = is_FerroArray(a)
    B = is_FerroArray(b)
    C = is_FerroArray(c)
    D = is_FerroArray(d)
    sum = A + B + C + D
    sum = sum/4
    return sum

#Makes a parametrization of the elements which I first discovered in a talk of Stefano Sanvito.
def load_den_param(formulas, data):
    A,B,C,D = giveAtomicNumber(formulas)
    unA = np.unique(A)
    unB =np.unique(B)
    unC=np.unique(C)
    unD=np.unique(D)
    unA=np.append(unA,unB)
    unA=np.append(unA,unC)
    unA=np.append(unA,unD)
    unA=np.unique(unA)
    lenMax=len(unA)
    DenParams = np.zeros((len(formulas), lenMax))
    unEl, fracs,err = load_fractions_array(data)
    unFormula=np.zeros(0,dtype='U100')
    for i in range (0,len(unEl)):
        unFormula=np.append(unFormula,str(unEl[i,0])+str(unEl[i,1])+str(unEl[i,2])+str(unEl[i,3]))
    A,B,C,D=giveAtomicNumber(unFormula)
    aN1=np.zeros(0)
    aN2=np.zeros(0)
    aN3=np.zeros(0)
    aN4=np.zeros(0)
    fra1 = fracs[:, 0]
    fra2 = fracs[:, 1]
    fra3 = fracs[:, 2]
    fra4 = fracs[:, 3]
    aN1=A
    aN2=B
    aN3=C
    aN4=D
    for i in range(len(fracs[:, 0])):
        if fra1[i]!=0: DenParams[i, np.where(unA == aN1[i])[0]] = fra1[i]
        if fra2[i]!=0: DenParams[i, np.where(unA == aN2[i])[0]] = fra2[i]
        if fra3[i]!=0: DenParams[i, np.where(unA == aN3[i])[0]] = fra3[i]
        if fra4[i]!=0: DenParams[i, np.where(unA == aN4[i])[0]] = fra4[i]
    return DenParams, unA,err

#Similar function as load_fractions but this function also loads the absolute magnetism as well as the summed magnetic moment of each element.
#This is done for each compound i in its jth occurence.
def load_fractions_mag(data, i,j):
    S,err=returnValues('systemSites-S',data,j,i)
    O,errN=returnValues('systemSites-O',data,j,i)
    err = errHandl(err, errN)
    M,errN=returnValues('systemSites-M',data,j,i)
    err = errHandl(err, errN)
    L,errN=returnValues('systemSites-L',data,j,i)
    err = errHandl(err, errN)
    S=S.astype(int)
    M=M.astype(float)
    unique = isolateUniqueEle(L)
    numUnique = len(unique)
    sumArray = np.zeros(0)
    absArray = np.zeros(0)
    fractionsArray = np.zeros(0)
    for i in range(0, numUnique):
        indicesArr = np.argwhere(L == unique[i])
        fractionsArray = np.append(fractionsArray, np.sum(np.array(O[indicesArr],dtype=float)))
        sumArray = np.append(sumArray, np.sum(M[indicesArr].astype(float)*O[indicesArr].astype(float)))
        absArray = np.append(absArray, np.sum(np.abs(M[indicesArr].astype(float))*O[indicesArr].astype(float)))
    return unique, fractionsArray, sumArray, absArray,err

#Sums the results of the percious function up to an array format from all compounds.
def load_fractions_array_mag(data):
    err=np.zeros(0)
    keysA=data.keys()
    unique = np.zeros((dataSize(data), 4),dtype='|S5')
    fractionsArray = np.zeros((dataSize(data), 4))
    sums = np.zeros((dataSize(data), 4))
    absArr = np.zeros((dataSize(data), 4))
    k=0
    for i in keysA:
        try:
            u, f, s, a,errN = load_fractions_mag(data, i,0)
            unique[k, 0:len(u)] = u
            fractionsArray[k, 0:len(u)] = f
            sums[k, 0:len(u)] = s
            absArr[k, 0:len(u)] = a
            err=np.append(err,errN)
            if np.size(errN)>0:
                unique[k, :] = 0
                fractionsArray[k, :] = 0
                sums[k, :] = 0
                absArr[k, :] = 0
                errN=[]
        except:
            err=np.append(err,i)
            unique[k,:]=0
            fractionsArray[k,:]=0
            sums[k,:]=0
            absArr[k,:]=0
        k=k+1
    for i in keysA:
        for j in range (1,jReturn(i,data)):
            try:
                u, f, s, a,errN = load_fractions_mag(data, i,j)
                unique[k, 0:len(u)] = u
                fractionsArray[k, 0:len(u)] = f
                sums[k, 0:len(u)] = s
                absArr[k, 0:len(u)] = a
                err=np.append(err,errN)
                if errN.size>0:
                    unique[k, :] = 0
                    fractionsArray[k, :] = 0
                    sums[k, :] = 0
                    absArr[k, :] = 0
                    errN=[]
            except:
                err=np.append(err,i+str(j))
                unique[k,:]=0
                fractionsArray[k,:]=0
                sums[k,:]=0
                absArr[k,:]=0
            k=k+1
    return unique, fractionsArray, sums, absArr, err

#Extracts the type of magnetism from the dataset  and encodes it as string of integers similar to the stochiometry. 
#It encodes the 'type' of magnetism as well as its strength given by the prefixes in the single data set. 
def classify_mag(singleDataSet):
    resStr = ''
    if np.any(singleDataSet == 'F'):
        resStr = resStr + '9'
    elif np.any(singleDataSet == 'iF'):
        resStr = resStr + '6'
    elif np.any(singleDataSet == 'sF'):
        resStr = resStr + '3'
    else:
        resStr = resStr + '0'
    if np.any(singleDataSet == 'A'):
        resStr = resStr + '9'
    elif np.any(singleDataSet == 'iA'):
        resStr = resStr + '6'
    elif np.any(singleDataSet == 'sA'):
        resStr = resStr + '3'
    else:
        resStr = resStr + '0'
    if np.any(singleDataSet == 'SS'):
        resStr = resStr + '9'
    elif np.any(singleDataSet == 'iSS'):
        resStr = resStr + '6'
    elif np.any(singleDataSet == 'sSS'):
        resStr = resStr + '3'
    else:
        resStr = resStr + '0'
    return resStr

#Loads all magnetic states from the whole dataset. 
def load_magnetic_state(data):
    recData,err = load_single_element('magnetic_state', data)
    state = np.zeros(0, dtype=str)
    for i in range(0, len(recData[:])):
        state = np.append(state, classify_mag(recData[i]))
    return state,err

#Loads the magnetic descripors from the data set and returns them preformatted. 
def load_magnetism(data):
    el, frac, sum, abs,err = load_fractions_array_mag(data)
    totalM = np.zeros(0)
    absM = np.zeros(0)
    for i in range(0, len(sum[:, 0])):
        totalM = np.append(totalM,np.sum(sum[i, :]))
        absM = np.append(absM, np.sum(abs[i, :]))
    mTot = totalM
    mAbs = absM
    m1 = sum[:, 0]
    m2 = sum[:, 1]
    m3 = sum[:, 2]
    m4 = sum[:, 3]
    absM1 = abs[:, 0]
    absM2 = abs[:, 1]
    absM3 = abs[:, 2]
    absM4 = abs[:, 3]
    state = load_magnetic_state(data)
    return mTot, mAbs, m1, m2, m3, m4, absM1, absM2, absM3, absM4, state, err

#Each unique symmetry gets assignes an integer value which is returned for each compound.
def load_sym_strings_encoded(data):
    labels, err = load_single_element('label', data) 
    uniqueLab = np.unique(labels)
    intArray = np.zeros(0,dtype=int)
    for i in range(0, len(labels)):
        intArray = np.append(intArray,np.where(labels[i] == uniqueLab[:]))
    return intArray, err

#Loads the Heuslers data from the named .json file. This file is available under: https://archive.materialscloud.org/record/2022.28
#Loads the descripors provided by the previous functions, performs most of the error handling and composes a list of strings which contains the descripor names. 
def loadHeuslersData():
    file = open("JuHemd.json")
    err=np.zeros(0)
    qStringsData = np.zeros(0, dtype=str)
    dataFrame = np.zeros(0)
    data = json.load(file)
    data_to_be_loaded_simply = ['formula', 'lattice_constant', 'etotal']
    for i in data_to_be_loaded_simply:
        dataFrame, qStringsData,errN = load_and_append_single_element(i, data, qStringsData, dataFrame)
        err = errHandl(err, errN)
    syStr,errN = load_sym_strings_encoded(data)
    err=errHandl(err,errN)
    dataFrame, qStringsData = append_single_element('Symmetry Code', syStr, qStringsData, dataFrame)
    mtot, mabs, M1, M2, M3, M4, absm1, absm2, absm3, absm4, state,errNe = load_magnetism(data)
    state=state[0]
    err = errHandl(err, errNe)
    dataFrame, qStringsData = append_single_element('mTot', mtot, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('mAbs', mabs, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('M1', M1, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('M2', M2, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('M3', M3, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('M4', M4, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('AbsM1', absm1, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('AbsM2', absm2, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('AbsM3', absm3, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('AbsM4', absm4, qStringsData, dataFrame)
    dataFrame, qStringsData = append_single_element('Magn State', state, qStringsData, dataFrame)
    stoch,errN = load_stochiometry(data)
    err = errHandl(err, errN)
    dataFrame, qStringsData = append_single_element('Stochiometry', stoch, qStringsData, dataFrame)
    DenPar, atomNumb,errN = load_den_param(dataFrame[:, 0], data)
    err = errHandl(err, errN)
    for k in range(0, len(DenPar[0, :])):
        dataFrame, qStringsData = append_single_element('Density Param for atom # ' + str(int(atomNumb[k])), DenPar[:, k],qStringsData, dataFrame)
    ferroConc = load_Ferro_conc(dataFrame[:, 0], data)
    dataFrame, qStringsData = append_single_element('Ferro Density', ferroConc, qStringsData, dataFrame)
    rareConc = load_conc_rare_earth(dataFrame[:, 0], data)
    dataFrame, qStringsData= append_single_element('Rare earth Materials Density', rareConc, qStringsData, dataFrame)
    aN1,aN2,aN3,aN4 = giveAtomicNumber(dataFrame[:, 0])
    dataFrame, qStringsData = append_single_element('Atomic Numbers of Atom 1', aN1, qStringsData,dataFrame)
    dataFrame, qStringsData = append_single_element('Atomic Numbers of Atom 2', aN2, qStringsData,dataFrame)
    dataFrame, qStringsData = append_single_element('Atomic Numbers of Atom 3', aN3, qStringsData,dataFrame)
    dataFrame, qStringsData = append_single_element('Atomic Numbers of Atom 4', aN4, qStringsData,dataFrame)
    return dataFrame, qStringsData, data,err

#Loads Tc from the database
def loadTc(dataName):
    Tc = np.zeros(0)
    Tc,err = load_single_element('Tc', dataName)
    return Tc,err

#Radomizes the data and the tc so clustering is avoided. 
def randomize_data(data, tc):
    Rdata, Rtc = sk.utils.shuffle(data, tc)
    return Rdata, Rtc

#Cleans data from all coumpounds which returned an error. Also cleans data from compounds with vanishing abolute magnetic moment (smaller than 0.1) and such with Tc=0. Calls randomization in the End
def clean_Data(data,tc,errArr,strings):
    for i in range (0,len(errArr)):
        errArr[i]=errArr[i].replace('_','')
    indArr=np.zeros(0,dtype=int)
    for i in range (0,len(data[:,0])):
        if not np.any(errArr==data[i,0]): indArr=np.append(indArr,int(i))
    dataClean=data[indArr,:]
    tcClean=tc[indArr]
    errArr=np.zeros(0,dtype=int)
    for i in range (0,len(tcClean)):
        if (not (tcClean[i]==0.0)) and float(data[int(i),np.argwhere(strings=='mAbs')])>0.1:
            errArr=np.append(errArr,int(i))
    dataClean=dataClean[errArr,:]
    tcClean=tcClean[errArr]
    dataRand,tcRand=randomize_data(dataClean,tcClean)
    return dataRand,tcRand

#Adds atomistic descripors to the data array as well as descripor names for each atom to the descriptor list. 
def addAtomData(dataFrame, stringsH):
    aN1, aN2, aN3, aN4 = giveAtomicNumber(dataFrame[:, 0])
    arrAtom,qstr=initAtomData()
    quantitiesArr=[3,7,8,9,11,12,13,14,15,20]
    for j in quantitiesArr:
        total=np.zeros(0,dtype=float)
        for i in range(0, 4):
            atomNum = i + 1
            quant=np.zeros(0,dtype=float)
            if i==0: arrAt=aN1
            if i==1: arrAt=aN2
            if i==2: arrAt=aN3
            if i==3: arrAt=aN4
            for k in range (0,len(aN1)):
                quant=np.append(quant,float(arrAtom[int(arrAt[k]),j]))
            dataFrame,stringsH=append_single_element(qstr[j] + ' of Atom ' + str(atomNum), quant, stringsH, dataFrame)
            if i==0:
                total=quant
            else:
                total=total+quant
            if i==3:
                dataFrame,stringsH=append_single_element(qstr[j] + 'TOTAL', total, stringsH, dataFrame)
    for i in range (0,4):
        if i == 0: arrAt = aN1
        if i == 1: arrAt = aN2
        if i == 2: arrAt = aN3
        if i == 3: arrAt = aN4
        vE=valenceEl(arrAt)
        if i==0: tot=vE
        else: tot=tot+vE
        dataFrame, stringsH = append_single_element('# of Valence electrons' + ' of Atom ' + str(i+1), vE, stringsH, dataFrame)
    dataFrame, stringsH = append_single_element('TOTAL # of Valence electrons', tot, stringsH, dataFrame)
    return dataFrame, stringsH

#Writes formatted and ML-suitable data to .txt file. 
def writeData(data, Tc, optStr=''):
    np.savetxt("Data" + optStr + ".txt", data, fmt="%s")
    np.savetxt("Tc" + optStr + ".txt", Tc, fmt="%s")
    return

#Writes a human readable descriptor array. 
def writeDescrArray(descrArr, optStr=""):
    np.savetxt('Descriptors' + optStr + '.txt', descrArr, fmt="%s")
    return

#Removes all DFT generated results from the data array. 
def clearFromDFTResluts(data, strings):
    DFTresults = ["mTot", "mAbs", "M1", "M2", "M3", "M4", "AbsM1", "AbsM2", "AbsM3", "AbsM4", "Magn State", "etotal",
                  "mtot"]
    indArray = np.zeros(0,dtype=int)

    for i in range(0, len(strings)):
        if not (strings[i] in DFTresults): indArray = np.append(indArray,i)
    data = data[:,  indArray]
    strings = strings[ indArray]
    return data, strings


#Execute the functions in correct order: 
data, strings, dataName,err = loadHeuslersData()
dataH,stringsH=addAtomData(data, strings)
Tc,errN = loadTc(dataName)
err = errHandl(err, errN)
dataH,Tc=clean_Data(dataH,Tc,err,strings)
writeData(dataH, Tc)
writeDescrArray(stringsH)
dataRed, stringsRed = clearFromDFTResluts(dataH, stringsH)
writeData(dataRed, Tc, 'ClearedFromDTF')
writeDescrArray(stringsRed, 'ClearedFromDFT')
