import pandas as pd
import numpy as np
import seaborn as sns
import math
import statistics
import random as rnd
from sklearn.utils import shuffle 
import math
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
def lin(trainx,trainy,theta):
    alpha=0.3
    t=[]
    hj=[]
    for i in range(0,5000):
        predicted=np.dot(trainx,theta)
        for k in range(0,len(trainx)):
            for l in range(0,len(trainx[0])):
                if l==0:
                    theta[l][0]=theta[l][0]-(((alpha)*((1/(1+2.71828**-predicted[k][0]))-trainy[k]))/len(trainx))
                else:
                    theta[l][0]=(theta[l][0]-(((((alpha)*((1/(1+2.71828**-predicted[k][0]))-trainy[k]))*(trainx[k][l])))/len(trainx)))
        error=0
        for k in range(0,len(trainx)):
            r=(1/(1+2.71828**-predicted[k][0]))
            if trainy[k]==1:
                error=error-math.log(r+0.0001)
            else:
                error=error-math.log(1.001-r)
        error=error/(2*len(trainx))
        t.append(error)
        hj.append(i)
    sns.lineplot(x=hj,y=t) 
    return(theta)
def check(testx,testy,theta):         
    prediction=np.dot(testx,theta)
    for i in range(len(test)):
        if prediction[i][0]>0.5:
            prediction[i][0]=1
        else:
            prediction[i][0]=0
    count1=0
    count0=0
    actual1=0
    actual0=0
    for i in range(len(test)):
        #print("prediction "+str(prediction[i][0])+" original "+str(testy[i]))
        if(testy[i]==1):
            actual1+=1
            if(prediction[i][0]==testy[i]):
                count1+=1
        else:
            actual0+=1
            if(prediction[i][0]==testy[i]):
                count0+=1
    cm=confusion_matrix(testy,prediction)
    print(cm)
    recall=cm[0][0]/(cm[0][0]+cm[1][0])
    precision=cm[0][0]/(cm[0][0]+cm[0][1])
    accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    print("Accuracy :: %.2f " % accuracy)
    print("Recall :: %.2f " % recall)
    print("Precision :: %.2f " % precision)
    print("F1-Score :: %.2f " % (2*((precision*recall)/(precision+recall))))
    auc = roc_auc_score(testy, prediction)
    print('AUC :: %.2f' % auc)
# =============================================================================
#     fpr, tpr, thresholds = roc_curve(testy, prediction)
#     plt.plot(fpr, tpr, color='red', label='ROC')
#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.show()
# =============================================================================
    return(prediction)
def scale(df):
    l=["OfficeVisits","Narcotics","DaysSinceLastERVisit","Pain","TotalVisits","ProviderCount","MedicalClaims","ClaimLines","AcuteDrugGapSmall"]
    for i in l:
        df[i]=(df[i]-df[i].mean())/(df[i].max()-df[i].min())
    return(df)
df=pd.read_csv("Copy of medical_care_quality.csv")
df100=pd.DataFrame({"theta":1},index=np.arange(len(df)))
df=pd.merge(df100,df,on=df.index)
df=df.drop(["key_0","MemberID","InpatientDays","ERVisits"],axis=1)
df["S0"]=0
df["S1"]=0
df["S0"].loc[df["StartedOnCombination"]==False]=1
df["S1"].loc[df["StartedOnCombination"]==True]=1
df=df.drop(["StartedOnCombination"],axis=1)
df=scale(df)
df=shuffle(df)
train=df.sample(frac=0.7,random_state=rnd.randint(0,133))
test=df.drop(train.index)
trainx=train.drop(["PoorCare"],axis=1)
trainy=train["PoorCare"]
testx=test.drop(["PoorCare"],axis=1)
testx1=testx
testy=test["PoorCare"]
trainx=np.array(trainx)
trainy=np.array(trainy)
testx=np.array(testx)
testy=np.array(testy)
print(df.info())
theta=[]
for i in range(len(df.columns)-1):
    x=rnd.random()
    theta.append([x])
theta=np.array(theta)
theta=lin(trainx,trainy,theta)
print(theta)
prediction=check(testx,testy,theta)
testx1["PoorCare"]=prediction

    