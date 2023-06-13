# Developer: Anthony Patregnani

# Course: LIS4930, Exploration into AI, Machine and Deep Learning

# Semester: Summer 2023

print("\nProgram Requirements:\n"
      + "1. Get \n"
      + "2. Clean\n"
      + "3. Prepare\n"
      + "4. Analyze\n"
      + "5. Display/Visualize\n")

# 1. import pandas (open source data analysis/science and AI package)

import pandas as pd

# Get Data - read cleaned pickle file

# 2. read cleaned pickle file from Assignment 2 into "mortality_data" variable 

mortality_data = pd.read_pickle('mortality_cleaned.pkl')

## Examine Cleaned Data

# 3. display first and last 5 rows with one command!
# Note: the following way *only* works in Jupyter Notebooks
# mortality_data

print(mortality_data)

## Display DataFrame Attributes

# 4. Use Pandas DataFrame info() Method:
# Prints:
# 1. number of columns
# 2. column labels
# 3. column data types
# 4. column usage
# 5. range Index
# 6. number of cells in each column (non-null values)

mortality_data.info()

# 5. Print the following datafram attributes: index, columns, size, and shape

print("Index:  ", mortality_data.index)
print("Columns:", mortality_data.columns)
print("Size:   ", mortality_data.size)
print("Shape:  ", mortality_data.shape)

# 6. display first 3 records sorted by DeathRate in ascending order 
# Note: sort_values() function sorts in ascending order by default

print(mortality_data.sort_values('DeathRate').head(3))

# 7. display last 3 records sorted by DeathRate in descending order

print(mortality_data.sort_values('DeathRate', ascending=False).head(3))

# 8. display first 3 records sorted by Year and DeathRate both in ascending order
# Note: major/minor order (left to right). Here, first Year, then DeathRate

print(mortality_data.sort_values(['Year', 'DeathRate']).head(3))

# 9. display first 3 records sorted by DeathRate and Year both in ascending order
# Note: major/minor order 

print(mortality_data.sort_values(['DeathRate', 'Year']).head(3))

# 10. display first 5 records sorted (first) by Year (ascending order), then DeathRate (descending order)

print(mortality_data.sort_values(['Year', 'DeathRate'], ascending=[True, False]).head())

# 11. display first 5 records sorted (first) by DeathRate (descending order), then Year (ascending order)

print(mortality_data.sort_values(['DeathRate', 'Year'], ascending=[False, True]).head())

# Preparing Data - apply Statistical Functions

# 12. Display DeathRate mean

print(mortality_data['DeathRate'].mean())

# 13. display concatenated string and numeric output

print("DeathRate mean: " + str(mortality_data.DeathRate.mean()))

# 14. Same as above, though formatted to two decimal places

print("DeathRate mean: " + str(format(mortality_data.DeathRate.mean(), ".2f")))

# 15. Display max AgeGroup and DeathRate values (two columns)
# Note: AgeGroup sorted by *ASCII* values (5>1)--that is, why it is important to zero-fill, when necessary (see below)

print(mortality_data[['AgeGroup', 'DeathRate']].max())

# 16. display total numbers (all columns, excluding null values)

print(mortality_data.count())

# 17. using pandas quantile() function, display 50% value of Year column, verify using median() function

print(mortality_data['Year'].quantile(0.5))
print(mortality_data['Year'].median())

# 18. same as above, though, display 50% value of both Year and DeathRate columns, verify using median() function

print(mortality_data[['Year', 'DeathRate']].quantile([.5]))
print(mortality_data[['Year', 'DeathRate']].median())

# 19. same as above, though, display 10% value of both Year and DeathRate columns, then 90% value of each, using one statement
# quantile values (Example: 10% and 90%)
# 10% year = 1911.5, and 10% deathrate less than 21.50
# 90% year = 2006.5, and 90% deathrate less than 430.85

print(mortality_data.quantile([.1,.9], numeric_only=True))

# 20. display cumlative sum of deathrate values (i.e, each row added to previous row), use cumsum() function

print(mortality_data['DeathRate'].cumsum())

# 21. verify cumsum() function by displaying first 5 records of DeathRate col

print(mortality_data['DeathRate'].head(5))

## Preparing data - column arithmetic

# 22. create and display two new columns (i.e., 'Mean' and
# 'Mean Centered' displays how far from the mean each value
# 'centering' refers to the process of subtracting the mean
# 'centering' is important when interpreting group effects of
# 'centering' redefines 'e point' for 'predictor' to be whatever
# Note: Explicit Line joining using backslash (\).
# https://docs.python.org/3/reference/Lexical analysis.html#

mortality_data[ 'Mean'] = mortality_data.DeathRate.mean()
mortality_data[ 'MeanCentered'] = \
mortality_data.DeathRate - mortality_data.DeathRate.mean

# 23. print mortality_data DF
#Note: new cols in mortality_data DataFrame
print(mortality_data)

## Preparing data - modify string data in column

# 24. modify col names, so that col names are treated 'equalLy, that is, when being sorted (here, using dictionary)
# inplace set to true to in order to replace values in DataFrame
mortality_data.AgeGroup.replace(
{'1-4 Years': '01-04 Years', '5-9 Years': '05-09 Years'},
inplace=True)

## Save prepped DataFrame to pickle file, then read

# 25. save prepped DataFrame to pickle file
mortality_data.to_pickle('mortality_prepped.pkl')

# 26. Read prepped DataFrame from pickle file, and print first 5 records
mortality_data = p.read_pickle ('mortality_prepped.pkl')
mortality_data.head()

## Shape data - using index (sometimes, makes plotting easier!)

## Set and use an index

# 27. display first 2 records (note current indexes)

mortality_data.head(2)

# 28. index "year" col, then display first two records (note new indexes)
# ***Note:*** indexed columns *cannot* be used as parameters in other methods/functions!

mortality_data = mortality_data.set_index('Year')
mortality_data.head(2)

# 29. reset original index, then display first two records (note original indexes)
# inplace set to true to in order to replace values in DataFrame

mortality_data.reset_index(inplace=True)
mortality_data.head(2)

# 30. create multi-index, check validity (i.e., no duplicates), then display first two records (note new indexes)
# set multi-index to 'Year' and 'AgeGroup'

mortality_data = mortality_data.set_index(['Year', 'AgeGroup '], verify_integrity=True)
mortality_data.head(2)

# 31. reset original index (using inplace), then display first two records (note new indexes)

mortality_data.reset_index(inplace=True)
mortality_data. head(2)

## Shape data - using pivot (wide data)
# • Similar to spreadsheet
# • Each observation defined by both
# 1. value at cell in table, and
# 2. coordinates of cell with respect to row and column indices.
# • Not: Cannot access variables in dataset by name
# • https://seaborn.pydata.org/tutorial/data_structure.htm|#long-form-vs-wide-form-data
## Pandas pivot() function:
# https://pandas.pydata.org/docs/user_guide/reshaping.htm|freshaping-pivot
# https://pandas.pydata.org/docs/reference/api/pandas.Dataframe.pivot.html
# *Parameters:*
# • index: Column to use to make new DataFrame's index. If None, uses existing index.
# • columns: Column to use to make new DataFrame's columns.
# • values: Column(s) to use for populating new DataFrame's values. If not specified, all remaining columns used
# • returns: Reshaped DataFrame
#  Exception: ValueError raised if there are any duplicate indexes (cannot reshape if index/column pair not unique!)

# 32. pivot the following DataFrame: index= 'Year', columns= 'AgeGroup*
# index: 'Year'
# column to use to make new frame's columns: 'AgeGroup'
# values: all remaining columns to populate new DataFrame's values
# Note: if no 'values' par, uses *all* remaining cols!
# assign resulting DataFrame to 'mortality_wide' variable, then display first 3 records

mortality_wide = mortality_data.pivot (index= "Year", columns= "AgeGroup")
mortality_wide.head(3)

# 33. pivot the following DataFrame:
# index: 'Year'
# column to use to make new frame's columns: 'AgeGroup'
# values: 'DeathRate'
# assign resulting DataFrame to 'mortality_wide' variable, then display first 3 records

mortality_wide = mortality_data.pivot (index= "Year", columns="AgeGroup", values="DeathRate")
mortality_wide.head(3)

# 34. save wide DataFrame to Excel file ('mortality_wide.xLsx')
# Note: *Be careful* saving wide DataFrame to Excel file changes index- -must change back (see below)!

mortality_wide.to_excel('mortality_wide.xlsx')

# 35. read saved Excel file and assign to 'mortality_wide' variable, then display first 4 records

mortality_wide = pd.read_excel ('mortality _wide.xlsx')
print(mortality_wide.head(4))

# 36. save wide DataFrame to pickle file ('mortality_wide.pkL')

mortality_wide. to_pickle('mortality_wide.pkl')

# 37. read saved pickle file and assign to 'mortality_wide' variable, then display first 5 records

mortality_wide = p.read_pickle ('mortality_wide.pkl')
mortality_wide.head()

## Shape data - using melt (long data)
# - Each variable is a column
# - Each observation is a row
# - can access variables in dataset by name
# - https://seaborn.pydata.org/tutorial/data_structure.html#long-form-vs-wide-form-data
## Pandas melt) function:
# https://pandas.pydata.org/docs/user_guide/reshaping.html#freshaping-by-melt
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html
# *Parameters:*
# • id vars: Column(s) to use as identifier variables.
# • value_vars: Column(s) to unpivot. If not specified, uses all columns not set as id_vars.
# • var_name: Name to use for 'variable' column. If None, uses frame.columns.name or 'variable'
# • value_name: Name to use for 'value' column.
# • col_level: If columns Multilndex, then use this level to melt.
# • ignore_index: If True, original index ignored. If False, original index retained. Index labels will be repeated as necessary.
# • Returns: Reshaped DataFrame

# 38. melt following DataFrame: 'mortality_wide'
# id_vars='Year'
# value_vars=['01-04 Years', '05-09 Years']
# var_name='AgeGroup'
# value_name='DeathRate'
# assign resulting DataFrame to 'mortality_Long' variable, then display first 4 records

mortality_long = mortality_wide.melt(id _vars='Year', value_vars=['01-04 Years', '05-09 Years'], var_name='AgeGroup', value_name='DeathRate')
mortality_long.head(4)

# 39. using pandas option_context() function, display 6 rows (first 3 and Last 3) and all cols
# Note: option_context() executes codeblock with set of options that revert to prior settings after execution
# Note: 'None' value for 'display. max_columns' property returns *all* cols.
with pd. option_context (
'display. max_rows', 6,
'display.max_columns', None):
    print(mortality_long)

## Analyze grouped data using (single) aggregate functions

# 40. display first 5 records of 'mortality_data' DF (i.e., prior to aggregation)

mortality_data.head()

# 41. group 'AgeGroup' by mean, and display DF

mortality_data.groupby('AgeGroup').mean()

# 42. group 'Year' by median, and display first 4 records
# Note: See numeric_onLy=True note above.

mortality_data.groupby('Year').median(numeric_only=True).head(4)

# 43. group 'Year' and 'AgeGroup' by count, and display first 5 records
# Note: only for demo purposes, since there is only 1 record for each group

mortality_data.groupby(['Year', 'AgeGroup']).count().head()

# 44. group 'AgeGroup', then return statistical summary for 'DeathRate' using 'describe()' function

mortality_data.groupby('AgeGroup')['DeathRate'].describe()

## Analyze grouped data using (multiple) aggregate functions

# 45. group 'AgeGroup', then using agg() function, aggregate *all* columns using 'mean' and 'median' functions

mortality_data.groupby('AgeGroup').agg(['mean', 'median'])

# 46. group 'AgeGroup', then using agg() function, aggregate onLy 'DeathRate' coLumn
# using 'mean, median, standard deviation, and 'unique' functions

mortality_data.groupby('AgeGroup')['DeathRate'].agg(['mean', 'median','std', 'nunique'])

# 47. group 'Year', then using agg() function, aggregate onLy 'DeathRate' coLumn
# using 'mean, median, standard deviation, min, max, variance, and 'unique' functions

mortality_data.groupby('Year')['DeathRate'].agg(['mean', 'median','std', 'min', 'max', 'var', 'nunique'])

## Visualize data

# Note: There is no "best" way to visualize data. Different questions are best answered by different visualizations!
# Chart Visualization: https://pandas.pydata.org/docs/user_guide/visualization.html
# An Intuitive Guide to Data Visualization in Python: https://www.analyticsvidhya.com/blog/2021/02/an-intuitive-guide-to-visualization-in-python/

# 48. pivot 'mortality_data' DataFrame using the following parameters:
# index: 'Year '
# column to use to make new frame's columns: 'AgeGroup'
# values: all remaining columns to populate new DataFrame's values
# Note: if no 'values' par, uses *all* remaining cols!
# after pivoting DataFrame, use plot() function to display Line plot (default)

mortality_data.pivot(index='Year', columns='AgeGroup')['DeathRate'].plot()

# 49. dispLay same plot as above by re-indexing mortality_wide DF
# Note: Because, saving wide DataFrame to Excel file changes index--must re -index (see above) !
# Change index back to 'Year'

mortality_wide = mortality_wide.set_index('Year')

# save wide DataFrame to pickle file ('mortality_wide. pkl')
# read saved pickle file and assign to 'mortality_wide' variable,
# then dispLay first 5 records to dispLay new index

mortality_wide.to_pickle(' mortality_wide. pkl')
mortality_wide.pd.read_pickle('mortality_wide.pkl')
mortality_wide.head()

## Outliers
# • Defined: "Unusual" dataset values
# • Problematic: Distort findings
# • Identitying outliers (simple):
# 1. Sort data
# 2. Graphs (e.g., boxplot (very easy to see!), scatterplot, and histogram)
# • Fixing:
# 1. Keep (not always bad!)
# 2. Drop
# 3. Modify (Winsorizing): https://en.wikipedia.org/wiki/Winsorizing
# A. Cap (values)
# B. Reassign (e.g., to mean/median value, or linear interpolation - "curve fitting"/estimating new values)
# C. Other (nonexclusive) procedures--that is, not permitting "exclusive" procedures like trimming or truncating values


# https://www.simpLypsychology.org/boxplots.html
# 50. using 'mortality_data' DataFrame, create a box plot (aka "box and whisker plot") for each age group
# Note: box plot displays five-number summary: min, Ist quartile, median (2nd quartile), 3rd quartile, and max.
# Note: interquartile range (IQR) is the "box"-that is, bet. Ist and 3rd quartile
# Note: outliers are *actual* min/max values (data points located outside whiskers of box plot)

mortality_wide.plot.box()

# 51. display Line plot for re-indexed mortality_wide DF
# Note: xlabel: 'Year', ylabel: 'Deaths per 100,000', title: 'DeathRate by AgeGroup'

mortality_wide.plot(xlabel='Year', label='Deaths per 100,000', title= 'DeathRate by AgeGroup')

# 52. using 'mortality_data' DataFrame, group 'AgeGroup', then aggregate only 'DeathRate' column
# using 'mean,' 'median,' and 'standard deviation' aggregate functions, then, plot vertical bar plot (rotate x-axis Labels 45 degrees)

mortality_data.groupby('AgeGroup')['DeathRate'].agg(['mean', 'median','std']).plot.bar(rot=45)

# 53. same as above using horizontal bar plot

mortality_data.groupby('AgeGroup')['DeathRate'].agg(['mean', 'median', 'std']).plot.barh()

# 54. using 'mortality_data' DataFrame, query the following years: 1900, 1925, 1950, 1975, 2000, group by 'Year'
# then aggregate only 'DeathRate' column using sum() function, and, plot the following graph using pie plot

mortality_data.query('Year in (1900, 1925, 1950, 1975, 2000)').groupby('Year').DeathRate.sum().plot.pie()

# 55. using 'mortality_wide' DataFrame,
# query the following years: 1900, 1925, 1950, 1975, 2000
# then, create subplots for each of the following age groups: 01-04, 05-09, 10-14, 15-19
# create a 2x2 Layout, set sharey par. to True (to turn off y labels on remaining horizontal charts), use figure size of 8"5" (WxH)

mortality_wide.query('Year in (1900, 1925, 1950, 1975, 2000)').plot.barh(
    title= ['Child Mortality: 01-94', 'Child Mortality: 05-09', 'Child Mortality: 10-14', 'Child Mortality: 15-19'],
    sharey=True, legend=False, subplots=True, layout=(2, 2), figsize= (8,5))

# https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.DataFrame.plot.scatter.htr
# 56. using 'mortality_data' DataFrame, query the following AgeGroup: 01-04 Years,
# then, plot the following graph using scatter plot, with a color of your choice
# Note, if receiving a color warning, just hit Shift+Enter in this cell
# Appears to be an Anaconda issue, as plot documentation is above.

mortality_data.query('AgeGroup == "01-04"').plot.scatter(x='AgeGroup', y='DeathRate', color='blue')