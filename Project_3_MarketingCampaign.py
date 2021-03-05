'''
File              : Project_3_MarketingCampaign.py
Name              : Senthilraj Srirangan
Date              : 02/17/2021
Assignment Number : 10.2 Project 3: Project Check-In/Milestone 2
Course            : DSC 680 - Applied Data Science
Exercise Details  :
    Predict whether the client will subscribe to a term deposit or not.
'''

# Import Libraries

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

#1. Import the CSV data file for EDA

pd.options.display.max_columns = None
#pd.options.display.max_rows = None
data = pd.read_csv('bank.csv')


# 2. Inspect the Data
print(data.head())
print(data.isnull().sum())
print(data.info())
print(data.shape)
print(data.describe(include='all'))


# 3. data conversion ,change all the yes and no to 1 and 0.

col = ['default','housing','loan','y']
# function definition
def convert(x):
    return x.map({'yes':1,'no':0})

# calling the function
data[col] = data[col].apply(convert)
print(data)


# Make Dummy variables for all the categorical Variables.
categorical = data.select_dtypes(include=['object'])
print(categorical.head())

# dummy variables of all categorical columns
dummies = pd.get_dummies(categorical,drop_first=True)
print(dummies.head())


# concatination of two dataframes 'bank' and 'dummies'
data = pd.concat([data,dummies],axis=1)
data.drop(columns=categorical.columns,axis=1,inplace=True)
print(data.head())
print(data.shape)
print(data.info())

# collecting all the continuous valued columns in a dataframe
check_out = data[['age','balance','day','duration','campaign','pdays','previous']]
check_out.head()


# Checking outliers

# collecting all the continuous valued columns in a dataframe
check_out = data[['age','balance','day','duration','campaign','pdays','previous']]
print('Checking outliers\n',check_out.head())


# creating boxplots for all the continuous columns of the dataframe
plt.figure(figsize=(15,10))
plt.subplot(2,4,1)
sns.boxplot(y='age',data=data)
plt.subplot(2,4,2)
sns.boxplot(y='balance',data=data)
plt.subplot(2,4,3)
sns.boxplot(y='day',data=data)
plt.subplot(2,4,4)
sns.boxplot(y='duration',data=data)
plt.subplot(2,4,5)
sns.boxplot(y='campaign',data=data)
plt.subplot(2,4,6)
sns.boxplot(y='pdays',data=data)
plt.subplot(2,4,7)
sns.boxplot(y='previous',data=data)
plt.show()


# Making a heatmap to find correlation
plt.figure(figsize=(40,30))
sns.heatmap(data.corr(),annot=True)
plt.show()

# Calculate the existing subscription rate
round((sum(data['y'])/len(data.index))*100,2)

# Splitting the target variable and the predictor features in two different dataframes from Train Test Split

# X will have all the features
X = data.drop(['y'],1)
# Y will have the target variable
Y = data['y']
print(X.head())
print(Y.head())


# 3. Train Test Split

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)


# 4. feature Scaling

scaler = StandardScaler()

x_train[['age','balance','day','duration','campaign','pdays','previous']] = scaler.fit_transform(x_train[['age','balance','day','duration','campaign','pdays','previous']])
print(x_train.head())


# 5. Model Building

# logistic regression model
logm = sm.GLM(y_train,(sm.add_constant(x_train)),family = sm.families.Binomial())
print(logm.fit().summary())

# 6. Feature Selection using RFE

logreg = LogisticRegression()

# Running RFE with 13 variables as output
rfe = RFE(logreg,20)
rfe = rfe.fit(x_train,y_train)

ls = list(zip(x_train.columns,rfe.support_,rfe.ranking_))
print(ls)
col1 = x_train.columns[rfe.support_]
print(col1)

# Assessing the model with StatsModels

x_train_sm = sm.add_constant(x_train[col1])
logm1 = sm.GLM(y_train,x_train_sm, family=sm.families.Binomial())
res = logm1.fit()
res.summary()


# predicted values of the train dataset giving the probability
y_train_pred = res.predict(x_train_sm)
print(y_train_pred[:10])

y_train_pred = y_train_pred.values.reshape(-1)
print(y_train_pred[:10])


# Creating the dataframe with the actual subscription flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Sub':y_train.values,'Sub_prob':y_train_pred})
y_train_pred_final['Cust_id'] = y_train.index
print(y_train_pred_final.head(10))


#Creating new column 'predict' with 1 if Sub_prob > 0.5 else 0
y_train_pred_final['predict'] = y_train_pred_final['Sub_prob'].map(lambda x: 1 if x>0.5 else 0)
y_train_pred_final.head(10)

# creating confusion matrix for the following prediction
confusion = metrics.confusion_matrix(y_train_pred_final.Sub, y_train_pred_final.predict)
print(confusion)

print(metrics.accuracy_score(y_train_pred_final.Sub, y_train_pred_final.predict))


# Creating StatsModel for checking p-values
x_train_sm = sm.add_constant(x_train[col1])
logm2 = sm.GLM(y_train,x_train_sm, family=sm.families.Binomial())
res = logm2.fit()
print(res.summary())

# The abobe p-values looks pretty good with all the values except the month_spt which has little more than it compared.
# dropping column 'month_sep'
col1 = col1.drop('month_sep',1)

# Again creating StatsModel for checking p-values
x_train_sm = sm.add_constant(x_train[col1])
logm2 = sm.GLM(y_train,x_train_sm, family=sm.families.Binomial())
res = logm2.fit()
print(res.summary())

# After dropping the column check the model
# predicting the probability once again after dropping the features from data
y_train_pred = res.predict(x_train_sm)
print(y_train_pred[:10])

# 8.Making predictions on Test data

# transforming the test data
x_test[['age','balance','day','duration','campaign','pdays','previous']] = scaler.transform(x_test[['age','balance','day','duration','campaign','pdays','previous']])
x_test = x_test[col1]
print(x_test.head())

x_test_sm = sm.add_constant(x_test)

# Making predictions on the test data¶

y_test_pred = res.predict(x_test_sm)
print(y_test_pred[:10])

# Converting y_test_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
print(y_pred_1.head())

# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

# Putting Cust_id to index
y_test_df['Cust_id'] = y_test_df.index


# Removing index for both dataframes to append them side by side
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_pred_1
y_pred = pd.concat([y_test_df,y_pred_1],axis=1)
print(y_pred.head())

# renaming the columns
y_pred.rename(columns={'y':'Sub',0:'Sub_prob'},inplace=True)
print(y_pred.head())

# putting the limit of 0.25 from the precision_recall_curve
y_pred['final_predict'] = y_pred.Sub_prob.map(lambda x: 1 if x>0.25 else 0)
print(y_pred.head())

y_train_pred_final['predict'] = y_train_pred_final['Sub_prob'].map(lambda x: 1 if x>0.5 else 0)
print(y_train_pred_final.head(10))

# overall accuracy of the model on test data
print(metrics.accuracy_score(y_pred.Sub, y_pred.final_predict))
