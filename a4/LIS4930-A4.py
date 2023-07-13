# Regression Analysis
## What is regression analysis?
### 1. Equation predicts "unknown values" (ie, dependent variables) based upon one or more known values (ie, independent voriable)
### 2. Dependent variables (also called]: response, outcome/output, or larger variables (respond to changes in (another variables)
### 3. Independent variables (also called): predictor, input, regressor, or explanatory variable(s) (predict/explain changed values of dependent variable(si
### 4. Goal: Find the best fitting line which can accurately predict the output.
## Dependent variables (output on y-axis) are always the ones being studied- that is, whose variations) is/are being modified someho
## Independent variables (input on x-axis) are always the ones being manipulated, to study and compare the effects on the dependent variabl
## Note: The designations independent and dependent variables are used to not imply "cause and effect" (as do "predictor" c "explanatory terms)
## Note: Based on the number of input and output variables, linear regression has three types:
### 1. Simple linear regression (1 DV/1 IV)
### 2. Multiple linear regression (1 DM/2 or more IVS)
### 3. Multivariate linear regression (2 or more DVs/2 or more (Vs)
### Simple linear regression: Only one independent variable affecting one dependent variable
### Multiple linear regression: Two or more independent variables affecting one dependent variable.
### Multivariate linear regression: Two or more independant variables affecting two or more dependent variables

# Independent Variables vs. Dependent Variables
## Independent Variables (predictors):
## • Can the variable(s) be manipulated or controlled?
## Does the variable(s) come before the other variables chronologically?
## • Is/are the variables being used to see the affects on another variable(s)?
## Dependent variables (outcomes):
## • Is/are the variable(s) being used as (a) measured outcome(s)?
## Does the variable(s) depend upon an(other) variable(s)?
## • Is/are this/these variable(s) measured after another variable(s) is/are modified?

# Assignment 4 - Predicitive Analysis (Simple linear regression)

## Developer: Anthony Patregnani

## Course: LIS4930, Exploration into AI, Machine and Deep Learning

## Semester: Summer 2023

print("\nProgram Requirements:\n"
      + "1. Contrast similarities/differences among AI vs. Machine Learning vs. Deep Learning \n"
      + "2. Identify correlations\n"
      + "3. Use Seaborn (Data visualization library)\n"
      + "4. Graph Correlations\n"
      + "5. Use simple linear regression\n"
      + "6. Create linear model\n"
      + "7. Plot regression line\n"
      + "8. Make predictions - using simple linear regression model\n"
      + "9. Plot Residuals\n")

# 1. import pandas (open source data analysis/science and AI package) and numpy to perform mathematical functions
import pandas as pd
import numpy as np

# Get data - read cleaned .csv file

# 2. assign cleaned .csv file to "advertising_data" variable, then read .csv file to "housing" variable
# read the given .csv file and view some sample records
advertising_data = pd.read_csv("my_company_data.csv")

# display first and last 5 records
advertising_data

# Advertising Dataset

# 3. print number of rows and columns
advertising_data.shape

# 4. print dataframe info (note: also indicates null values, which, if present, would need to be remedied.)
advertising_data.info()

# 5. print dataframe statistics summary
advertising_data.describe()

# 6. Display pairwise correlations of *all* columns in the dataframe.
advertising_data.corr().head()

# note: *Perfect* correlations (1.0) with same attributes (ex TV with TV)
# Note: high correlation between TV and sales, much less with radio

# 7. import matplotlib and seaborn libraries to visualize data
import matplotlib.pyplot as plt
import seaborn as sns

# 8. visualize data for correlations using pairplot(), y=DV(s) and x=IV(s)
# pairplot(): Plot pairwise relationships in dataset
# Note: Scatter plot good when comparing two numeric variables, like here!
sns.pairplot(advertising_data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

# 9. Display one attribute's correlation ("price") to *all* other columns in dataframe, sorted in descending order by price.
advertising_data.corr()[['Sales']].sort_values(by='Sales', ascending=False)

# 10. Focus on correlation between TV and Sales
sns.relplot(data=advertising_data, x='TV', y='Sales')

# Note: Seaborn relational plot (relplot) visualizes how variables relate to each other within a dataset
# Note: when looking for correlation, see if a line can be drawn through as many data points as possible

# 11. visually display correlations using Seaborn's heatmap() function
sns.heatmap(advertising_data.corr(), cmap="YlGnBu", annot = True)
plt.show()

# 12. Visually condense correlations of one variables to other variables.
sns.heatmap(data=advertising_data.corr()[['Sales']].sort_values(by='Sales', ascending=False), annot=True, cmap='YlGnBu', cbar=False, fmt=f'.2f')

# Note: *annot* property set to *True* so that r-values are displayed in each cell


# Simple Linear Regression Steps

## 1. Identify x (IV) and y (DV): x=TV, y=Sales
## 2. Create Train and Test Datasets
## 3. Train model
## 4. Evaluate model

# 13. Identify x (IV) and y (DV)
x = advertising_data['TV']
y = advertising_data['Sales']

# 14. Create Train and test datasets
# a. split variables into training and testing sets
# b. build model using training set, then run model on testing set

# split variables into train and test datasets into 7:3 ratio, respectively, by importing train_test_split
# 70% of observations for training, and 30% for testing

# random_state: controls randomization during splitting
# random_state value not important-- can be any non-negative integer

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = 100)

# Review Train datasets after splitting

# 15. Display x training dataset
x_train

# 16. Display y training dataset
y_train

## Add column to perform regression fit properly for simple linear regression:

# 17. Display shape for x train and test datasets, *before* adding column
print(x_train.shape)
print(x_test.shape)

# Prepare Model

# 18. Add additional column to train and test datasets
# By reshaping we can add or remove dimensions
# reshape() concept: first, raveling array
# then inserting/deleting elements from reveled array into new array, using same index order for reveling
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

# 19. display shape for x train and test data, *after* adding column 
print(x_train.shape)
print(x_test.shape)

# Fitting model: Find best fitting line -- to accurately predict output

# 20. fit regression line to plot
from sklearn.linear_model import LinearRegression

ln = LinearRegression().fit(x_train, y_train)

# Intercept and Slope - Why?

# print model coefficients

# 21. print intercept value: Represents mean value of DV when all of IV(s) in model are equal to zero (also known as y-intercept)
print("Intercept :", ln.intercept_)

# 22. print slope value: Represents how much DV expected to change, as IV(s) increases/decreases
print("Slope :", ln.coef_)

# 23. simple linear regression formula from above values
# Y = mX + b
# Y = DV, X = IV, m = estimated slope, b = estimated intercept

# r2 values - how well regression line approximates actual data

# 24. predict y_value
y_train_predict = ln.predict(x_train)
y_test_predict = ln.predict(x_test)

from sklearn.metrics import r2_score

# 25. print and compare r2 values of both train and test data
print(r2_score(y_train, y_train_predict))
print(r2_score(y_test, y_test_predict))

# Note: (r2): Statistical measure of how well regression line approximates actual data
# Represents the proportion of variance for a DV that's explained by IV(s) in a regression model
# Bottom-line: r2 values on test data within 5% of r2 values on training data. Model looks good!

# Plot Linear regression (using simple estimation)

# 26. Plot data and linear model fit
sns.regplot(data=advertising_data, x='TV', y='Sales', ci=85, scatter_kws={'s':5}, line_kws={'lw':1, 'color':'orange'})
# scatter_kws changes dot size, line_kws changes line color and width
# ci (confidence interval used for regression estimate)

# Plot residuals using Seaborn

# 27. Residual plot: used to plot residual values after plotting linear regression model.
sns.residplot(data=advertising_data, x='TV', y='Sales', scatter_kws={'s':5}, lowess=True, line_kws={'color':'green'})

