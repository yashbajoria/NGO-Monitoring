import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
import pickle


d =pd.read_csv('newdc.csv')

X=d.iloc[: , 1:14].values


y=d.iloc[:,15].values

#d=d[['networth(lac.)','follower','victims reach(k)','national award','long for(in year)','staff']]


d.head()

#X=d.iloc[:,:].values

#Applying feature scaling....
"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
"""
#X[:, 1]=X[:, 1]/100

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)



x_show_train=X_train
x_show_test=X_test




from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #This uses normal equation to calcuate theta for minimum cost function
regressor.fit(X_train, y_train)

Y=regressor.predict(X_test)
#y_pred=regressorTest.predict(x_trial_test)

plt.scatter(x_show_test[:, 0], y_test,color='red')
plt.plot(x_show_test[:, 0], Y)
plt.show()

j=(np.sqrt(metrics.mean_squared_error(y_test, Y)))
#checking for coorelation matrix of dataset
#which give idea of relationship among features


#Creating .pkl file....
pickle.dump(regressor, open('ngo.pkl', 'wb'))

#Loading the model....
ngo=pickle.load(open('ngo.pkl', 'rb'));
#print(simple_linear_regression.predict())


"""
#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)



import matplotlib.pyplot as plt
plt.figure(figsize=(16,16))
cor = d.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#on the basis of correlation matrix we filter out unrelatable features
X_test[0]

from sklearn.decomposition import PCA
pca=PCA(n_components=1)
x_trial_train=pca.fit_transform(X_train)
x_trial_test=pca.fit_transform(X_test)



from sklearn.preprocessing import PolynomialFeatures
regressorTest=PolynomialFeatures(degree=1)
x_temp=regressorTest.fit_transform(x_trial_train)
x_show=x_trial_test;
x_trial_test=regressorTest.fit_transform(x_trial_test)
pol=LinearRegression()
pol.fit(x_temp, y_train)
y_show=pol.predict(x_trial_test)
#regressorTest.fit(x_temp, y_train)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X_train=poly_reg.fit_transform(X_train)
X_test=poly_reg.fit_transform(X_test)


jt=(np.sqrt(metrics.mean_squared_error(y_test, y_show)))

yyy=regressor.predict([X_test[0]])
def getvalue(entry):
    res=regressor.predict([entry])
    return res

def checkReport(lis):
    k=lis[12]
    if k>=3:
        ans=getvalue(lis)
        return ans
    elif k==2:
        s="HIGH"
        printf(s)
        return
    elif k==1:
        s="MODERATE"
        print(s)
        return
    elif k==0:
        s="LOW"
        print(s)
        return
    else:
        s="ERROR"
        print(s)
        return


o=getvalue(lis)
t=checkReport(X_test[0])
        





"""

        








