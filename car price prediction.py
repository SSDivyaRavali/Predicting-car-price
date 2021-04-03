import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math
from sklearn import linear_model,metrics
import seaborn as sns
from scipy import stats
from datetime import datetime
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.metrics import r2_score
import datetime
import time
from datetime import date,timedelta
sns.set(style="white")
df=pd.read_csv('C:/Users/acer/vs/cars_sample.csv', sep=',', encoding='Latin1')
#to get non null objects
#df.info()
#to get total null objects
#pd.isnull(df).sum()
missing_cols=["vehicleType","gearbox","model","fuelType","notRepairedDamage"]
categorical_cols=["vehicleType","gearbox","model","fuelType","notRepairedDamage","seller","offerType","abtest","brand"]
numeric_cols=["price","yearOfRegistration","powerPS","kilometer","monthOfRegistration","postalCode"]
df["vehicleType"].fillna("others", inplace=True)
df["notRepairedDamage"].fillna("Not available",inplace=True)
df["gearbox"].fillna("UnSpecified", inplace=True)
df["model"].fillna("Not Available", inplace=True)
df["fuelType"].fillna("Not Specified", inplace=True)
#inspecting yearOfregistration and price
#(df.describe())
df=df[(df["powerPS"]<=1000)&(df["powerPS"]>0)]
df = df[(df["price"] >=100.0)]
df = df[(df["yearOfRegistration"] >= 1885) & (df["yearOfRegistration"] <2016)]
df = df.loc[(df.kilometer>1000)&(df.kilometer<=150000)]
# Mean of the prices of all the vehicle types
_median = df.groupby("vehicleType")["price"].median()
# 75th percentile of the prices of all the vehicles types
_quantile75 = df.groupby("vehicleType")["price"].quantile(0.75)
# 25th percentile of the prices of all the vehicles types
_quantile25 = df.groupby("vehicleType")["price"].quantile(0.25)
# Calculating the value of the prices of each vehicle type above which all the values are outliers
iqr_upper = ((_quantile75 - _quantile25)*1.5+_quantile75)
iqr_lower=(_quantile25-(_quantile75-_quantile25)*1.5)
#removing outliers
df = df[((df["vehicleType"] == "station wagon") & (df["price"] <= 17525.0)&(df["price"]>=-8275.0)) |
        ((df["vehicleType"] == "others") & (df["price"] <= 11751.5)&(df["price"]>=-5452.5)) |
        ((df["vehicleType"] == "suv") & (df["price"] <= 36375.0)&(df["price"]>=-13025.0)) |
        ((df["vehicleType"] == "bus") & (df["price"] <= 18450.0)&(df["price"]>=-7550.0)) |
        ((df["vehicleType"] == "cabrio") & (df["price"] <= 28001.5)&(df["price"]>=-12002.5)) |
        ((df["vehicleType"] == "limousine") & (df["price"] <= 17651.5)&(df["price"]>=-8352.5)) |
        ((df["vehicleType"] == "coupe") & (df["price"] <= 34475.0)&(df["price"]>=-17725.0)) |
        ((df["vehicleType"] == "Not Specified") & (df["price"] <= 5225.0)&(df["price"]>=-2783.0)) |
        ((df["vehicleType"] == "small car") & (df["price"] <= 7850.0)&(df["price"]>=-3510.0))]      

# calculating no of years the vehicle is old#Replacing all the 0 month values to 1
df.monthOfRegistration.replace(0,1,inplace=True)
# Making the year and month column to get a single date
Purchase_Datetime=pd.to_datetime({'year':df.yearOfRegistration,'month':df.monthOfRegistration,'day':[1]*len(df)})
df["Purchase_date"]=Purchase_Datetime
now=datetime.datetime.now()
today=now.date()
print(today)
# Calculating days old by subracting both date fields and converting them into integer
Days_old=(now-Purchase_Datetime).dt.days
#print(Days_old)
#type(Days_old[1])
df['Days_old']=Days_old
df= df.drop(['dateCreated','lastSeen','Purchase_date','monthOfRegistration','name','dateCrawled','yearOfRegistration',
                 'offerType','abtest','seller','postalCode',],axis=1)  
Cat_feature=['notRepairedDamage','vehicleType','model','brand','gearbox','fuelType']
le= LabelEncoder()
cat= df.columns[df.dtypes==object]
for col in cat:
    df[col]= le.fit_transform(df[col])
#Skweness lead to non linearity
#print(df["price"].describe())
print("skewness: %f" % df['price'].skew())
print("kurtosis: %f" % df['price'].kurt())
df["price_new"]=np.log(df["price"].values)

print("skewness after log: %f" % df['price_new'].skew())
#df.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
corr=df.corr()["price_new"]
print(corr[np.argsort(corr, axis=0)[::-1]])
#y=df['price_new']
#X=df.drop(['price', 'price_new'], axis=1)  
cars_dummies=pd.get_dummies(data=df,columns=['notRepairedDamage','vehicleType','model','brand','gearbox','fuelType'])
X = cars_dummies.drop(['price','price_new','notRepairedDamage','vehicleType','model','brand','gearbox','fuelType'],axis=1)    
y=cars_dummies.loc[:,cars_dummies.columns=='price_new']
print(X.head())
"""
#High Trainnig accuracy
linreg = linear_model.LinearRegression()
linreg.fit(X,y)
print (linreg.intercept_)
lin_coef=list(zip(X.columns.get_values(), linreg.coef_))  
"""
#otofsample accuracy  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=0)
linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)
coef=list(zip(X_train.columns.get_values(), linreg.coef_))
print('Intercept: ',linreg.intercept_)
y_pred = linreg.predict(X_test) 
y_test = np.asanyarray(y_test[['price_new']])
print("Residual sum of squares(MSE): %.2f" % np.mean((y_pred - y_test) ** 2))
MSE=np.mean((y_pred - y_test) ** 2)
print("Root mean squares error (RMSE): %.2f" % np.sqrt(MSE))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linreg.score(X_test, y_test))
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y_test)))
print("R2-score: %.2f" % metrics.r2_score(y_pred ,y_test))
print(linreg.score(X_test,y_test)*100,'% Prediction Accuracy')
#residualPlot 
fig=plt.figure(figsize=(10, 5))
plt.scatter(linreg.predict(X_train),linreg.predict(X_train)-y_train,c='b',s=40,alpha=0.5)
plt.scatter(linreg.predict(X_test),linreg.predict(X_test)-y_test,c='g',s=40)
plt.hlines(y=0,xmin=-20,xmax=20)
plt.title("Residual Plot using training(blue) and test(green) data")
plt.ylabel("residuals")
plt.xlabel("fitted value")
plt.show()
fig.savefig("C:/Users/acer/vs/residual_plot_cat.png")
#plotting predicted values
fig=plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual  Price')
plt.ylabel('Predicted Price')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout()
plt.show()
fig.savefig("C:/Users/acer/vs/Actual_pred.png")
#results = pd.DataFrame({'Actual_price':,'Predicted_price':[y_pred[:2]]}) 
#print(results.head())
