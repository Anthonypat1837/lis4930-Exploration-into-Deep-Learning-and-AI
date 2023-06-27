# Regression Analysis
## What is regression analysis?
### 1. Equation predicts "unksown values" (le, dependent variables) based upon one or more known values (ie, independent voriables)
### 2. Dependent variables (also called): response, oufcome/butpur, or larger variables (respond to changes in (another variables
### 3. Independent variables (also called): predictor, input, regressor, or explanatory variable(s) (predict/explain changed values of dependent variable(si
### 4. Goal: Find the best fitting line which can accurately predict the output.
## Dependent variables (output on y-axis) are afways the ones being studied - that is, , whose variations) is/are being modified somehow!
## independent variables (input on x-axis) are always the ones being manipulated, to study and compare the effects on the dependent var
## https://www.statisticssolutions.com/independent-and-dependent-variables/
## https://www.scribbe.com/methodologw/independent-and-dependent-variables/
## Note: The designations independent and dependent variables are used to not imply "cause and effect? (as do "predictor' or "explanatory terms).
## Note: Based on the number of input and output variables, linear regression has three types:
### 1. Simple linear regression (1 DV/1 I)
### 2. Multiple linear regression (1 DM/2 or more IV5)
### 3. Multivariate linear regression (2 or more OVS/2 or more (VS)
## Simple linear regression: Only one independent variable affecting one dependent variabl
## Multiple linear regression: Two or more independent variables affecting one dependent variable.

# Multiple linear regression: Two or more independent variables affecting one dependent variable.
# Multivariate linear regression: Two or more independent variables affecting two or more dependent variables.

# Linear Regression in Python:

# https://www.realpython.com/linear-regression-in-python/

# Linear Regression in 2 minutes:

# https://www.youtube.com/watch?v=CtsRRUddV2s 

# Regression in Machine Learning: What it is and examples of different models:

# https://www.builtin.com/data-science/regression-machine-learning

# Independent Variables vs. Dependent Variables

## Independent Variables (predictors):

### • Can the variable(s) be manipulated or controlled?

### • "Do(es) the variable(s) come before the other variables) chronologically?

### • Is/are the variables being used to see the affects on (anjother variable(s)?

## Dependent variables (outcomes):

### • Is/are the variable(s) being used as (a) measured outcome(st?

### • "Do(es) the variable(s) depend upon an(other) variable|s)?

### • Is/are this/these variables) measured after (anjother variable(s) is/are modified?

# Project 1 - Predictive Analysis (Simple Linear Regression)

## Developer: Anthony Patregnani

## Course: Exploration into Al, Machine and Deep Leaming

## Program Requirements:

# 1. Contrast similarities/differences among Al vs. Machine-Leaming vs. Deep-Leaming
# 2. Identify correlations
# 3. Use Seabom (data visualization library built on top of matplotlib)
# 4. Grach correlations
# 5. Use simple linear regression
# 6. Create linear model
# 7. Plot regression line
# &. Mace predictions - using simple linear regression model
# 9. Plot residuals


# 1. import pondas and seaborn (butlt on top of matplotith)

import pandas as pd
import seaborn as sns

# Get data - read cleaned .csv file

# 2. assign cleaned .csv file to "housing data" variable, then read .cv file to "houstng" variable
# Note: Here is a to-step process for demonstration purposes (tvo steps easter than using Long paths and/or Long file names, as per note below)

housing_data = "housing_data.csv" # *MUST* be in same directory as -ipynb file
housing = pd.read_csv(housing_data)

# Note: Could have asso accomplished the process in one step...
# housing = pd.read_csw("housting_data.csv") # *MUST* be in same directory as . ipynb file
# 3. find houses where sqft of Living space is Less than 8800 sqft and price is greater then $0, and Less than $1,000,009
housing = housing.query('sqft_living < 8000 and price < 1000000 and price > 0')

# 4. Add additional attribute to "housing" dataframe called "has_basement"--*if* basement SQFT is greater than 0.
housing['has_basement'] = housing['sqft_basement'].apply(lambda x: True if x > 0 else False)

# Note: Lambda expressions are used to construct anonymous functions- -using the "Lambda" keyword.
# pandas.Dataframe.apply() can be used to execute Lambda expressions. It also con take ony number of arguments.

# 5. drop unconcerned columns (Note: Other attributes could be fecLuded and studted Later.)
housing = housing.drop(columns=['date','street','city', 'statezip', 'country', 'sqft_lot', 'yr_renovated', 'sqft_basement'])

# Housing Dataset

# 6. print dataframe info 
# Note: also, indicates null values, which, if present, would need to be remedied.
housing.info()

# 7. print first 5 records (glimpse new dataframe)
housing.head()

# Identifying correlations using scatterplot

# 8. See if there is any correlation (small) between "price" and "living SQFT"
sns.relplot(data=housing, x='sqft_living', y='price')

# Note: Seaborn relational plot (relplot) visualizes how variables relate to each other within a dataset.

# 9. See if there is any correlation (essentially *none*) between "price" and year built
sns.relplot(data=housing, x='yr_built', y='price')

# Here, essentially, there is no line that can be drawn through any discernable "grouping".

# Identifying correlations using grid scatter plots

# 10. Faster way of displaying relationships between pairs of datapoints.

sns.pairplot(data=housing,
             y_vars=['price', 'sqft_living', 'sqft_above'],
             x_vars=['price', 'sqft_living', 'sqft_above'],
             diag_kind='kde')

# Note: By default, pairplot() creates a grid of axes
# That is, each numeric variable in a dataset shared across y-axis, across a single *row*; and the x-axis across a single *column*.
# Each cell plots corresponding x and y variables

# Identifying correlations with r-values
# (i.e., a measure of any linear trend between two variables)

# 11. Display pairwise correlations of *all* columns in the dataframe.
housing.corr().head()

# 12. Display one attribute's correlation ("price") to *all* other columns in dataframe, sorted in descending order by price.
housing.corr()[['price']].sort_values(by='price', ascending=False)

# Identify correlations with heatmaps

# 13. Visually display correlations using Seaborn's heatmap() function.
sns.heatmap(data=housing.corr(), cmap='Blues', vmin=1.0, vmax=1.0)

# Note: vmin/vmax properties determine color shading

# 14. Visually condense the correlations of one variable to other variables.
sns.heatmap(data=housing.corr()[['price']].sort_values(by='price', ascending=False), amnot=True, cmap='Blues', cbar=False, fmt=f'.2f')

# Note: amnot value set to *True* so that r-values are displayed in each cell.

# Create, validate and use simple linear regression model
## Steps:
### 1. Split dataset (training vs test datasets)
### 2. Create model from "training" dataset
### 3. Validate model using "test" dataset
### 4. If valid, predict values

# 15. import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

# Note: sklearn used for predictive data analysis 
# Built on top of NumPy, SciPy, and matplotlib

# i. split data (X: IV and Y: DV split into training and test datasets)

x_train, x_test, y_train, y_test = train_test_split(housing[['sqft_living']], housing[['price']], test_size=0.33, random_state=42)

# ii. create model from training dataset
linearModel = LinearRegression() # instantiates LinearRegression object
linearModel.fit(x_train, y_train) # accepts x,y training sets and fits regression line to dataset

# iii. validate model using test dataset
linearModel.score(x_test, y_test)

# Note: score() function accepts test dataset and returns R squared value for regression.
# R squared value is percent of change in DV attributable to to the IV(s). 

# iv. use model to make predictions on test data
# accepts IV "test" dataset and returns predicted DV "outcome" dataset values
y_predicted = linearModel.predict(x_test)
y_predicted

# Note: Model predicts output y-axis (DV) values, on basis of input x-axis (IV) values in x_test.


# Plot predicted data (preparation)

# 16. make dataframe of predicted prices
predicted = pd.DataFrame(y_predicted, columns=['price_predicted'])
predicted

# 17. combine test data and predicted data (reset index for test columns for accurate comparisons between test/predicted values)
combined = predicted.join([x_test.reset_index(drop=True), y_test.reset_index(drop=True)])
combined # Note: Predicted values are *not* highly correlated to actual values (again, roughly, only 35%)

# 18. melt price and price_predicted columns into single column
# Note: As mentioned, "sqft_living" is the IV. And, price_type indicates whether price is an actual or predicted price.
melted = pd.melt(combined, id_vars=['sqft_living'], value_vars=['price', 'price_predicted'], var_name='price_type', value_name='price_value')

# display first 5 records
melted.head()

# Plot predicted data

# 19. Plot text (actual) and training (predicted) data
sns.relplot(data=melted, x='sqft_living', y='price_value', hue='price_type')

# Note: "hue" parameter used to distinguish between actual and predicted values

# Plot linear regression (using simple estimation)

# 20. Similar to plot above; here, linear regression model automatically generated (simple "estimation").
sns.lmplot(data=housing, x='sqft_living', y='price', ci=None, scatter_kws={'s':5}, line_kws={"lw":1, 'color':'red'})

# Plot residuals - preparation

# 21. print dataframe summary to confine attribute names and data types
combined.info()

# 22. Plotting regression residuals can also be used to evaluate models, apart from the score() function
# Hre, residual values stored in new attribute in *combined* DF called "residual."
combined['residual'] = combined.price - combined.price_predicted
combined.head()

# Note: The difference between DV and IV values indicate variance in predictions.

# Plot residuals using Seaborn

# 23. Plot residuals using Seaborn relplot(). Here, x-axis is IV and y-axis are the residuals.
g = sns.relplot(data=combined, x='sqft_living', y='residual')

# Next, draw horizontal line where y-axis = 0 (i.e., prediction equals actual value -- in other words, prediction is correct).
for ax in g.axes.flat:
    ax.axhline(0, ls='--')

# 24. Similar to relplot (fitted model) vs implot (simple estimation), residplot is simple residual estimation
sns.residplot(data=housing, x='sqft_living', y='price', scatter_kws={'s':5})







