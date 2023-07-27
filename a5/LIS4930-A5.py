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


# Assignment 5 - Predicitive Analysis (Multiple linear regression)

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


# Import Libraries

# 1. import necessary libraries
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Get data - read cleaned .csv file

# 2. assign cleaned .csv file to "fish" variable
# read the given .csv file and view some sample records
fish = pd.read_csv("fish.csv")

# display first and last 5 records
fish

# Clean Data

# 3. Remove cols to differentiate "length" properties, then display first 5 records
fish.rename(columns={"Length1":"VerticalLength",
                     "Length2":"DiagonalLength",
                     "Length3":"CrossLength"}, inplace=True)
fish.head()

# Analyze Dataset

# 4. print number of rows and columns

fish.shape

# 5. print dataframe info
fish.info()

# 6. print dataframe statistics summary

fish_describe = fish.describe()

# format entire dataframe to two decimal places
pd.options.display.float_format = "{:,.2f}".format

print(fish_describe)

# Identify correlations

# 7. Display pairwise correlations of *all* columns in dataframe
fish.corr(numeric_only = True).head()

# 8. visualize data for correlations using pairplot(). y=DV(s), x=IV(s)
sns.pairplot(fish, x_vars=["Height", "Width", "VerticalLength"], y_vars="Weight", height=4, aspect=1, kind='scatter')
plt.show()

# 9. Custom Visuals
sns.color_palette()

# 10. Display color palette referenced by name
sns.color_palette("pastel")

# 11. display color values as hex codes
print(sns.color_palette("pastel").as_hex())

# 12. Custom Plotting
sns.pairplot(fish, hue="Species", diag_kind="hist", markers=["o", "s", "D", "*", "<", "p", "v"], palette="husl")

# 13. Display one attribute's correlation ("Weight") to all other columns in dataframe, sorted in descending order by Weight
fish.corr(numeric_only=True)[['Weight']].sort_values(by="Weight", ascending=False)

# 14. Viually display correlations using Seaborn's heatmap() function

ax=sns.heatmap(data=fish.corr(numeric_only=True), annot=True, cmap='viridis')
ax.tick_params(axis='both', rotation=45, labelsize=8, labelcolor='blue', color='green')

# 15. Visually condense correlations of one variable to other variables
sns.heatmap(data=fish.corr(numeric_only=True)[['Weight']].sort_values(by='Weight', ascending=False), annot=True, cmap="PuBuGn", fmt=f'.2f')

# Create Multiple regression model
## Multiple Regression steps
### 1. Identify x and y. x=height, width, verticallength, and y=weight
### 2. Create Train and Test Datasets
### 3. Train model
### 4. Evaluate Model

# 16. Focus on one species: Bream
bream = fish.query('Species == "Bream"')

# 17. Identify x and y
x = bream[['Height', 'Width', 'VerticalLength']]
y = bream[['Weight']]

# 18. Create train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.75, test_size=0.25, random_state= 100)

# 19. Display x training dataset (IVs)
x_train

# 20. Display y training dataset (DV)
y_train/454

# Fitting model: Find best fitting line -- to accurately predict output

# 21. fit regression line to plot
model = LinearRegression().fit(x_train, y_train)

# 22. display R2 correlation values -- validates model through correlation score
model.score(x_test, y_test)

# 23. Predict weight and display predicted values based upon IVs
# predict() method accepts x values from test dataset and returns predicted y values
y_predicted = model.predict(x_test)

# create "predicted" DataFrame with "PredictedWeight" column using y_predicted values
predicted = pd.DataFrame(y_predicted, columns = ['PredictedWeight'])

# display data structure and predicted weight 
predicted

# 24. display data structure type and values
x_test

# 25. display data structure type and values
y_test

# 26. Join and Display all columns
final = predicted.join([x_test.reset_index(drop=True),
                        y_test.reset_index(drop=True)])

# display PredictedWeight and actual weight
final

# Residuals 
## R2 value returned by score() method for test dataset provides good indication of regression model validity
## Also, plotting residuals help to evaluate the models

# 27. calculate and display residual values
# Note: Residuals are simply differences between DV test values and DV predicted values

final['Residuals'] = final.Weight - final.PredictedWeight

final

# 28. Plot residuals
# Plot reveals that outliers affecting regression are on negative side of curve
sns.kdeplot(data=final, x='Residuals')

