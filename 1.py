import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math
import sklearn
import seaborn as sns
from datetime import datetime
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

#df["monthOfRegistration"].replace([0,12],[1,11],inplace=True)"]
# calculating no of years the vehicle is old
df["yearsOld"] = 2016 - df["yearOfRegistration"]
# calculating no of months the vehicle is old
#df["monthsOld"] = 12 - df["monthOfRegistration"]

part=df[["yearOfRegistration","price","kilometer","powerPS","vehicleType","gearbox","fuelType","notRepairedDamage","brand","yearsOld"]]
#split data into train and test
msk = np.random.rand(len(df)) < 0.8
train = part[msk]
test = part[~msk]
#Train data distribution 
# scatterplot for price based on year of registration
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(x="powerPS", y="price",data=train,)
ax.set_title("Price of car based on power ",fontdict= {'size':12})
ax.xaxis.set_label_text("Power of car",fontdict= {'size':14})
ax.yaxis.set_label_text("Price",fontdict= {'size':14})
#simple regression model
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['kilometer']])
train_y = np.asanyarray(train[['price']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
#plot the fitted line
plt.scatter(train.kilometer, train.price,  color='blue')
plt.plot(train_x, (regr.coef_[0][0]*train_x + regr.intercept_[0]), '-r')
plt.xlabel("kilometer")
plt.ylabel("price")
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['kilometer']])
test_y = np.asanyarray(test[['price']])
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y))
