# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:53:31 2017

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import stats

#"\r" usually is interpreted as a special character and means carriage return. Either add a 'r' prefix 
#to your string literals which prevents this special sequence from being interpreted 
#(e.g. path = r"foo\rar")

train = pd.read_csv(r'E:\DataScience\PythonModelingPractice\BlackFriday\BlackFridayProject\train.csv')

test = pd.read_csv(r'E:\DataScience\PythonModelingPractice\BlackFriday\BlackFridayProject\test.csv')
test1 = test
#print (test.head())
#print(train.isnull().values.any())
#print (train.dtypes)
#print (train.info())
#print (train.isnull().sum())  ## Find number of NA in each colunm
#print (train.isnull().sum().sum())  # Total NA in data
#print (train['Product_Category_2'].median())  ## Find median of Product_Category_2 to fill NA
train['Product_Category_2'].fillna(value=9, inplace = True)  ## Filling NA for Product_Category_2
#print (train['Product_Category_3'].median())  ## Find median of Product_Category_2 to fill NA
train['Product_Category_3'].fillna(value=14, inplace=True)  ## Filling NA for Product_Category_2
#print (train.isnull().sum()) ## Find number of NA in each colunm
#print (train.isnull().sum().sum()) # Total NA in data
#print (test.isnull().sum())
#print (test['Product_Category_2'].median())
test['Product_Category_2'].fillna(value=9, inplace=True)
#print (test['Product_Category_3'].median())
test['Product_Category_3'].fillna(value=14,inplace=True)
#print (test.isnull().sum())
#print (test.isnull().sum().sum())
#print (train.head(3))
#print (train.shape)
## Number of Unique value in User ID. Both syntaxes below are correct
#print(train.User_ID.nunique())
#print (train['User_ID'].nunique())

## Number of unique in Product ID
#print (train.Product_ID.nunique())
#print ((train.dtypes == 'object').sum())  ## There are 5 categorical variables
#print (set(test.columns).difference(set(train.columns)))
#print ((train.dtypes == 'int64').sum())     # there are 4 variables which have integer as dtype
#print ((train.dtypes == 'float64').sum())  # there are 2 variables which have float as dtype 
  
# Check the distribution of dependent variable i.e Purchase     
((train['Purchase']).hist(bins=20).set_title('PurchasePattern'))

# Check the distributiton of Product_Category_1
train['Product_Category_1'].hist(bins=20).set_title('Product category 1 pattern')

# Check the distribution of Product_Category_2 
train['Product_Category_2'].hist(bins=20).set_title('Product Category 2 pattern')

# Check the distribution of Product_Category_3 
train['Product_Category_3'].hist(bins=20).set_title('Product Category 3 pattern')

# Check the distribution of Occupation
train['Occupation'].hist(bins=20).set_title('Occupation')

#print (train.dtypes=='object')

# plot bar chart for categorical variables
dim = (11,5)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=dim)
sns.countplot(x="Gender", data = train, palette="Greens_d", ax=ax1).set_title('Train data - Gender wise count')#More males are purchasing
sns.countplot(x="Gender", data = test, palette="Blues_d", ax=ax2).set_title('Test data - Gender wise count') #More males are purchasing

dim = (20,5)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True, figsize=dim)
#There are many count in Age 26-35 and very less in 0-17 in train & test data
sns.countplot(x="Age", data = train, palette = "Reds_d", ax=ax1).set_title('Train data - Age wise count')
sns.countplot(x='Age', data = test, palette = "Greens_d", ax = ax2).set_title('Test data - Age wise count')

## Number of count of City
dim = (11,5)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=dim)
#People from City B are purchasing more
sns.countplot(x='City_Category', data=train, palette="Reds", ax=ax1).set_title('Train data- City wise count')
sns.countplot(x='City_Category', data = test, palette="BuGn_r", ax=ax2).set_title('Test data- City wise count')

## Number of count of 'Stay_In_Current_City_Years'
dim = (11,5)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=dim)
#People who have stayed for 1 year are purching more compared to others
sns.countplot(x='Stay_In_Current_City_Years', data=train, palette='Greens', ax=ax1).set_title('Train data - Stay in Current City in Yrs')
sns.countplot(x='Stay_In_Current_City_Years', data=test, palette="Blues", ax=ax2).set_title('Test data - Stay in Current City in Yrs')

## Number of count of Marital_Status
dim = (11,5)
fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=True, figsize=dim)
sns.countplot(x='Marital_Status', data = train, palette="Blues", ax=ax1).set_title('Train data - Marital status count')
sns.countplot(x='Marital_Status', data = test, palette="Greens", ax=ax2).set_title('Test data - Marital status count')

## Number of count of Product_ID 
dim = (11,5)
fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=True, figsize= dim)
#sns.countplot(x='Product_ID', data = train, palette="Blues", ax=ax1).set_title('Train data - Product ID Count')
#sns.countplot(x='Product_ID', data = test, palette="BuGn_r", ax=ax2).set_title('Test data - Product ID Count')

gndr_grp = train.groupby('Gender')
print (gndr_grp['Purchase'].describe())

gndr_grp.boxplot(column='Purchase', return_type = 'axes')
# make purchase data for hypotheis of Female & Male to find out either F & M are 
#significantly different
M_Purchase = train[train['Gender']=='M']['Purchase']
F_Purchase = train[train['Gender']=='F']['Purchase']

#By doing Hypothesis we can say that they have mean purchase difference i.e they are significant
print (stats.ttest_ind(M_Purchase, F_Purchase, equal_var=False))

#Test the ANOVA
print(stats.f_oneway(M_Purchase, F_Purchase))

## Age wise hypothesis
age_grp = train.groupby('Age')
age_grp.boxplot(column='Purchase', return_type='axes', figsize=(10,10))
print (age_grp.describe())

age_grp1=train[train['Age']=='0-17']['Purchase']
age_grp2 = train[train['Age']=='18-25']['Purchase']
age_grp3=train[train['Age']=='26-35']['Purchase']
age_grp4=train[train['Age']=='36-45']['Purchase']
age_grp5=train[train['Age']=='46-50']['Purchase']
age_grp6=train[train['Age']=='51-55']['Purchase']
age_grp7=train[train['Age']=='55+']['Purchase']

#Test the ANOVA
#Age groups have different purchase mean,so we will consider this Age variable for model building
print(stats.f_oneway(age_grp1, age_grp2, age_grp3, age_grp4, age_grp5, age_grp6, age_grp7))

mar_grp = train.groupby('Marital_Status')
mar_grp.boxplot(column='Purchase', return_type = 'axes')
print (mar_grp.describe())

mar_grp1 = train[train['Marital_Status'] == 0]['Purchase']
mar_grp2 = train[train['Marital_Status'] ==1]['Purchase']
#Test the ANOVA
#Marital Status groups 1 & 0 are not significantly different from one another they have equal purchase mean 
#so while doing regression we will not consider this variable for model building.
print (stats.f_oneway(mar_grp1, mar_grp2))

city_grp = train.groupby('City_Category')
city_grp.boxplot(column='Purchase', return_type='axes')
print (city_grp.describe())

city_grp1 = train[train['City_Category'] == 'A']['Purchase']
city_grp2 = train[train['City_Category'] == 'B']['Purchase']
city_grp3 = train[train['City_Category'] =='C']['Purchase']

#Test the ANOVA
#City_Category variables groups have signifiacntly 
#different purchase mean, we can take this variable for analysis.
print (stats.f_oneway(city_grp1, city_grp2, city_grp3))

stay_grp = train.groupby('Stay_In_Current_City_Years')
stay_grp.boxplot(column='Purchase', return_type='axes')
print (stay_grp.describe())

stay_grp1 = train[train['Stay_In_Current_City_Years'] == '0']['Purchase']
stay_grp2 = train[train['Stay_In_Current_City_Years'] == '1']['Purchase']
stay_grp3 = train[train['Stay_In_Current_City_Years'] == '2']['Purchase']
stay_grp4 = train[train['Stay_In_Current_City_Years'] == '3']['Purchase']
stay_grp5 = train[train['Stay_In_Current_City_Years'] == '4+']['Purchase']

#Test the ANOVA
#Stay_In_Current_City_Years groups have same purchase mean difference
print (stats.f_oneway(stay_grp1,stay_grp2,stay_grp3,stay_grp4,stay_grp5))

prdId_grp = train.groupby('Product_ID')
#prdId_grp.boxplot(column='Purchase', return_type = 'axes')
print (prdId_grp.describe())

prdId_grp1 = train[train['Product_ID'] == 'P00000142']['Purchase']
prdId_grp2 = train[train['Product_ID'] == 'P00000242']['Purchase']
prdId_grp3 = train[train['Product_ID'] == 'P00000342']['Purchase']
prdId_grp4 = train[train['Product_ID'] == 'P00000442']['Purchase']
prdId_grp5 = train[train['Product_ID'] == 'P0099642']['Purchase']
prdId_grp6 = train[train['Product_ID'] == 'P0099742']['Purchase']
prdId_grp7 = train[train['Product_ID'] == 'P0099842']['Purchase']
prdId_grp8 = train[train['Product_ID'] == 'P0099942']['Purchase']

#Product ID groups have significant difference in purchase mean
print (stats.f_oneway(prdId_grp1,prdId_grp2,prdId_grp3,prdId_grp4,prdId_grp5,prdId_grp6,prdId_grp7,prdId_grp8))

# group Product_Category_1
pr_cat_grp1 = train.groupby('Product_Category_1')

pr_cat_grp1.boxplot(column = 'Purchase', return_type = 'axes')
print (pr_cat_grp1['Purchase'].describe())

#pr_cat_grp11 = train[train['Product_Category_1'] == 1]['Purchase']
#pr_cat_grp12 = train[train['Product_Category_1']==2]['Purchase']
#pr_cat_grp13 = train[train['Product_Category_1']==3]['Purchase']
#pr_cat_grp14 = train[train['Product_Category_1']==4]['Purchase']
#pr_cat_grp15 = train[train['Product_Category_1'] == 5]['Purchase']

#pr_cat_grp16 = train[train['Product_Category_1'] == 6]['Purchase']
#pr_cat_grp17 = train[train['Product_Category_1']==7]['Purchase']
#pr_cat_grp18 = train[train['Product_Category_1']==8]['Purchase']
#pr_cat_grp19 = train[train['Product_Category_1']==9]['Purchase']
#pr_cat_grp110 = train[train['Product_Category_1'] == 10]['Purchase']

#pr_cat_grp111 = train[train['Product_Category_1'] == 11]['Purchase']
#pr_cat_grp112 = train[train['Product_Category_1']==12]['Purchase']
#pr_cat_grp113 = train[train['Product_Category_1']==13]['Purchase']
#pr_cat_grp114 = train[train['Product_Category_1']==14]['Purchase']
#pr_cat_grp115 = train[train['Product_Category_1'] == 15]['Purchase']

#pr_cat_grp116 = train[train['Product_Category_1'] == 16]['Purchase']
#pr_cat_grp117 = train[train['Product_Category_1']==17]['Purchase']
#pr_cat_grp118 = train[train['Product_Category_1']==18]['Purchase']
#pr_cat_grp119 = train[train['Product_Category_1']==19]['Purchase']
#pr_cat_grp120 = train[train['Product_Category_1'] == 20]['Purchase']

#print (stats.f_oneway(pr_cat_grp11,pr_cat_grp12,pr_cat_grp13,pr_cat_grp14,pr_cat_grp15,pr_cat_grp16,
#               pr_cat_grp17,pr_cat_grp18,pr_cat_grp19,pr_cat_grp110,pr_cat_grp111,pr_cat_grp112,
#               pr_cat_grp113,pr_cat_grp114,pr_cat_grp115,pr_cat_grp116,pr_cat_grp117,pr_cat_grp118,
#               pr_cat_grp119,pr_cat_grp120))

pr_cat_grp2 = train.groupby('Product_Category_2')
print (pr_cat_grp2['Purchase'].describe())

pr_cat_grp3 = train.groupby('Product_Category_3')
print (pr_cat_grp3['Purchase'].describe())

occ_grp = train.groupby('Occupation')
print (occ_grp['Purchase'].describe())

impute_grps = train.pivot_table(values =['Purchase'], index = ['Gender', 'City_Category', 'Stay_In_Current_City_Years', 'Occupation'],
                                aggfunc=np.mean)
print (impute_grps)

## Stacked chart for visualization
var = train.groupby(['Age', 'Gender']).Gender.count()
var.unstack().plot(kind = 'bar', stacked = True, grid = False, figsize=(10,5))

## Chi-sq test
chi_AG = pd.crosstab(train['Gender'], train['Age'])
print (chi_AG)
#Gender & Age are dependent on each other
print(stats.chi2_contingency(chi_AG))

var1 = train.groupby(['City_Category', 'Gender']).Gender.count()
var1.unstack().plot(kind='bar', stacked=True, grid=False, figsize=(10,5))

#Chi-sq Test
chi_CG=pd.crosstab(train['City_Category'], train['Gender'])
print (chi_CG)
#Gender & City_Category dependent
print (stats.chi2_contingency(chi_CG))

var2=train.groupby(['Marital_Status', 'Gender']).Gender.count()
var2.unstack().plot(kind='bar', stacked=True, grid=False, figsize=(10,5), color=['grey','lightblue'])

chi_MG = pd.crosstab(train['Marital_Status'], train['Gender'])
print (stats.chi2_contingency(chi_MG))

plt.figure(figsize=(15,4))
sns.boxplot(train['Purchase'], color='c')

plt.figure(figsize=(15,6))
sns.distplot(train['Purchase'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data = train['User_ID'].append(test['User_ID'])

le.fit(data.values)

train['User_ID'] = le.transform(train['User_ID'])
test['User_ID'] = le.transform(test['User_ID'])

data1=train['Occupation'].append(test['Occupation'])

le.fit(data1.values)

train['Occupation'] = le.transform(train['Occupation'])
test['Occupation'] = le.transform(test['Occupation'])

#Label Encoding

train_x = train.ix[:,[0,1,2,3,4,5,6,7,8,9,10,11]] #for subseting by index
test_x = test.ix[:,[0,1,2,3,4,5,6,7,8,9,10]]

#Importing LabelEncoder and initializing it
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in train_x.columns.values:
    if train_x[col].dtypes == 'object':
        data = train_x[col].append(test_x[col])
        le.fit(data.values)
        train_x[col] = le.transform(train_x[col])
        test_x[col] = le.transform(test_x[col])
        
## Create dummy variables
train_x = pd.get_dummies(train_x, columns=['City_Category', 'Age', 'Stay_In_Current_City_Years'])
test_x = pd.get_dummies(test_x, columns=['City_Category', 'Age', 'Stay_In_Current_City_Years'])

print (train_x.shape)
print (test_x.shape)

train['User_ID'] = train['User_ID'].astype('category')
test['Product_ID'] = test['Product_ID'].astype('category')
train['Product_ID'] = train['Product_ID'].astype('category')
test['User_ID'] = test['User_ID'].astype('category')

## Split the data for to check the accuracy
from sklearn.cross_validation import train_test_split
x_train, x_test = train_test_split(train_x, test_size=0.3)

x_train_org = x_train
x_test_org = x_test

print (x_train.shape)
print (x_test.shape)
print (x_test.head())

#Linear Regression
#•	Without Dummy Variables
#•	With Dummy Variables
import statsmodels.formula.api as smf

# create a fitted model in one line
lm = smf.ols(formula='Purchase ~Age_0+Age_1+Age_2+Age_3+Age_4+Age_5+Age_6+Gender+Occupation+Product_ID+City_Category_0+City_Category_1+City_Category_2+City_Category_0+Product_Category_1+Stay_In_Current_City_Years_0+Stay_In_Current_City_Years_1+Stay_In_Current_City_Years_2+Stay_In_Current_City_Years_3+Stay_In_Current_City_Years_4+Product_Category_2+Product_Category_3', data=x_train).fit()
print (lm.summary())

pred = lm.predict(x_test)
MSE = np.mean((x_test['Purchase']-pred) **2)
print ("RMSE = %.2f" %np.sqrt(MSE))

#we will remove Stay_In_Current_City_Years variable because it is having p-value more 
#than 0.05 which says it is not a significant variable in our model.
## Removing Stay_In_Current_City_Years & Marital Status

# create a fitted model in one line
lm1 = smf.ols(formula='Purchase ~Age_0+Age_1+Age_2+Age_3+Age_4+Age_5+Age_6+Gender+Occupation+Product_ID+City_Category_0+City_Category_1+City_Category_0+Product_Category_1+Product_Category_2+Product_Category_3', data=x_train).fit()

print (lm1.summary())

pred0 = lm1.predict(x_train)
pred1 = lm1.predict(x_test)
MSE1 = np.mean((x_test['Purchase']-pred1) **2)
## No change in RMSE
print ("Calculated RMSE = %.2f" %np.sqrt(MSE1))

from statsmodels.graphics.regressionplots import plot_leverage_resid2
fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(lm1, ax=ax)

# Scatter plot the training data

train = plt.scatter(pred0, (x_train['Purchase']-pred0), c='b', alpha=0.5)

test_org = x_test['Purchase']-pred1
# Scatter plot the test data
test = plt.scatter(pred1, (x_test['Purchase']-pred1), c='r', alpha=0.5)

#plot horizontal axis line at 0
plt.hlines(y=0, xmin=-10, xmax=50)

plt.legend((train,test), ('Training', 'Testing'), loc='lower left')
plt.title('Residual Plots')

# create X and y
feature_cols = ['Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6','Gender','Occupation','Product_ID','City_Category_0','City_Category_1','City_Category_2','Product_Category_1','Product_Category_2','Product_Category_3']

X = train_x[feature_cols]
Y = train_x['Purchase']

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X,Y)

## Take only the feature used in model buliding
test_xp_org = x_test
test_xp = x_test[feature_cols]

# Predict on test
pred = lm.predict(test_xp)

# Save into DataFrame
output = pd.DataFrame(data={'User_ID':test_xp_org['User_ID'], 'Product_ID':test_xp_org['Product_ID'], 'Purchase':pred})
#output = pd.DataFrame(data={"User_ID":test1['User_ID']})
# Use pandas to write the comma-separated output file
output.to_csv('F:\BF.csv', index=False)

#Model with Dummy variables
lm_full = smf.ols(formula='Purchase ~Age_0+Age_1+Age_2+Age_3+Age_4+Age_5+Age_6+Gender+Occupation+Product_ID+City_Category_0+City_Category_1+City_Category_0+Product_Category_1+Product_Category_2+Product_Category_3', data=train_x).fit()
print (lm_full.summary())

# Predict on test
prediction = lm_full.predict(test_x)

# Save into DataFrame
output = pd.DataFrame( data={"User_ID":test_x["User_ID"], "Product_ID":test_x['Product_ID'],"Purchase":prediction} )

# Use pandas to write the comma-separated output file
output.to_csv( "F:\Reg_Model_dummy.csv", index=False)

#Decision Tree
feature = ['Age_0', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6', 'Gender', 'User_ID', 'Occupation', 'Product_ID', 'City_Category_0', 'City_Category_1', 'City_Category_2', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']

x = x_train[feature]
y = x_train['Purchase']

x_tst = x_test[feature]
y_tst = x_test['Purchase']

## full data subsetting
main_x = train_x[feature]
main_y = train_x['Purchase']
main_tst = test_x[feature]

from sklearn import tree
tree = tree.DecisionTreeRegressor(max_depth = 7)
model_tree = tree.fit(x,y)

pred_tree = model_tree.predict(x_tst)

MSE = np.mean((y_tst - pred_tree) **2)
print ("Training set RMSE using Tree dept=7, = %.2f" %np.sqrt(MSE))

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=7,n_estimators=100,verbose = 1)
model_rf = rf.fit(x,y)

pred_rf = model_rf.predict(x_tst)

MSE_RF = np.mean((y_tst - pred_rf)**2)
print ("Training set RMSE using RF depth=7, est=100, = %.2f" %np.sqrt(MSE_RF))

rf1 = RandomForestRegressor(max_depth=7, n_estimators=200)
model_rf1 = rf1.fit(main_x, main_y)

pred_rf1 = model_rf1.predict(main_tst)

output = pd.DataFrame(data={'User_ID':main_tst['User_ID'], 'Product_ID':main_tst['Product_ID'], 'Purchase':pred_rf1} )
                      
output.to_csv('F:\BF_RF', index=False)

x = x_train[feature]
y = x_train['Purchase']

x_tst = x_train[feature]
y_tst = x_train['Purchase']

tree_rf2 = RandomForestRegressor(max_depth=10, n_estimators=100)
model_tr2 = tree_rf2.fit(x,y)

pred_rf2 = model_tr2.predict(x_tst)

MSE_rf2 = np.mean((y_tst - pred_rf2)**2)

print ("Training set RMSE using dept=10, est=100, = %.2f" %np.sqrt(MSE_rf2))

# Full model fitting with Random forest
tree_rf3 = RandomForestRegressor(max_depth=10, n_estimators=100)
model_rf3 = tree_rf3.fit(main_x, main_y)

pred_rf3 = model_rf3.predict(main_tst)

output = pd.DataFrame(data = {'User_ID':main_tst['User_ID'], 'Product_ID': main_tst['Product_ID'], 'Purchase': pred_rf3})

output.to_csv('F:\RF_DP10_Full_Model.csv', index=False)

## We will import h2o and initialize it
import h2o
h2o.init(nthreads=-1)

from h2o.estimators.gbm import H2OGradientBoostingEstimator

## Data Preparation for h2o
feature = ['User_ID', 'Product_ID']
x=x_train
x_train_h2o = h2o.H2OFrame(x)

testx = x_test
x_test_h2o = h2o.H2OFrame(testx)

gbm_reg = H2OGradientBoostingEstimator(distribution='gaussian', nthreads=500, max_depth=3, learn_rate=0.04, nbins_cats=5891)
model_reg = gbm_reg.train(x=feature, y='Purchase', training_frame=x_train_h2o)

print (gbm_reg)
