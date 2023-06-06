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

# Get Data 

# 2. get mortality data: National Center for Health Statistics (NCHS childhood mortality rate)

mortality_url = "https://data.cdc.gov/api/views/v6ab-adf5/rows.csv"

# 3. read csv file into DataFrame

mortality_data = pd.read_csv(mortality_url)

# 4. save DF to pickle file

mortality_data.to_pickle('mortality_data.pkl')

# 5. read pickle file
# Note: Below, data in mortality_data variable changed--no longer csv data, now pickle data

mortality_data = pd.read_pickle('mortality_data.pkl')

# ways to print an entire Pandas DataFrame:
# https://www.geeksforgeeks.org/how-to-print-an-entire-pandas-dataframe-in-python/
# Or ...
# display Pandas DataFrame in table style:
# https://www.geeksforgeeks.org/display-the-pandas-dataframe-in-table-style/
# 6. Display first and last five rows with one command!
# mortality_data # Note: *only* works in Jupyter Notebooks
print(mortality_data)

# 7. first 5 rows only

print(mortality_data.head())

# 8. last 3 rows only
mortality_data.tail(3)

# 9. using pandas option_context() function, display 6 rows (first three and last three) and all cols (Note: using odd numbers doesn't appear to work)
# Note: option_context() - executes codeblock with set of options that revert to prior settings after excecution
# Note: 'None' value for 'display.max_columns' property returns *all* cols.

with pd.option_context(
    'display.max_rows', 6,
    'display.max_columns', None):
    print(mortality_data)

# 10. print follwoing DataFrame attributes: index, columns, size, and shape
print("Index:  ", mortality_data.index)
print("Columns:", mortality_data.columns)
print("Size:   ", mortality_data.size)
print("Shape:  ", mortality_data.shape)

# 11. display DataFrame values in array format

print(mortality_data.values)

# Use columns attribute to change column names (i.e., remove any spaces)

# 12. Use columns attribute, and replace() function to remove spaces in column names
# removing spaces in column names good practice (otherwise, processing commands limited)

mortality_data.columns = mortality_data.columns.str.replace(" ", "")

# 13. print column names

print(mortality_data.columns)

# 14. print column names with first 5 records

print(mortality_data.head())

# Use info(), nunique(), and describe() methods

# 15. print DataFrame information

mortality_data.info()

# 16. print DataFrame information, including accurate memory usage (using 'deep' value)

mortality_data.info(memory_usage='deep')

# 17. summarize unique values (i.e., no duplicates) in each column

mortality_data.nunique()

# 18. display generic stats for each numeric column - using describe() function

mortality_data.describe()

# 19. transpose stats so that stat names are display in columns (using T property)

mortality_data.describe().T

# 20. save cleaned DataFrame to pickle file (mortality_cleaned.pkl)

mortality_data.to_pickle('mortality_cleaned.pkl')

# 21. read cleaned pickle file to mortality_data variable

mortality_data = pd.read_pickle('mortality_cleaned.pkl')

# 22. display first 5 rows of cleaned pickle file

print(mortality_data.head())

# Accessing Data (Be Careful!)

# Access Data (columns)

# using dot notation
# Note: can't use dot notation with more than one column--or column names with spaces!
# 23. Display first two records of DeathRate column using dot notation.

print(mortality_data.DeathRate.head(2))

# 24. Same as above using single brackets

print(mortality_data['DeathRate'].head(2))

# 25. Same as above using double brackets

print(mortality_data[['DeathRate']].head(2))

# Note differences in types!
# 26. Display DeathRate type using dot notation.
# note type (Series)

print(type(mortality_data.DeathRate))

# 27. Display DeathRate type using single brackets.
# note type single brackets (Series)

print(type(mortality_data['DeathRate']))

# 28. Display DeathRate type using double brackets.
# note type double brackets (DataFrame)

print(type(mortality_data[['DeathRate']]))

# 29. Access more than one column--*must* use brackets

print(mortality_data[['Year','DeathRate']].head(2))

# 30. Display Year and DeathRate type using double brackets
# note type (DataFrame)

type(mortality_data[['Year','DeathRate']])

# Accessing Rows

# 31. Access Data using query() function (Year=1900)

print(mortality_data.query('Year==1900'))

# 32. Access Data using query() function (Year=2000 and AgeGroup != 1 - 4 years)

print(mortality_data.query('Year == 2000 and AgeGroup != "1-4 years"'))

# 33. Access Data using query() function (Year=2000 or Year=1900_, display first 5 records

print(mortality_data.query('Year == 1900 or Year == 2000').head())

# Access subset of rosw and columns

# Note: use backticks if a column name contains spaces
# mortality_data.query('Year == 200 and 'Age Group' !- "1-4 Years"')

# Note: after using query method on subset can use either dot notation or brackets
# Generally, use brackets, returns a DataFrame (easier to perform some operations), instead of Series object.

# 34. Using dot notation, access data using query() function (Year=1900), only for DeathRate column, display first 5 records

print(mortality_data.query('Year==1900').DeathRate.head())

# 35. Using single bracket notation, access data using query() function (Year=1900), only for DeathRate column, display first 5 records

print(mortality_data.query('Year==1900')['DeathRate'].head())

# 36. Using double bracket notation, access data using query() function (Year=1900), only for DeathRate column, display first 5 records

print(mortality_data.query('Year==1900')[['DeathRate']].head())

# 37. Display data type of dot notation used above

print(type(mortality_data.query('Year == 1900').DeathRate.head()))

# 38. Display data type of single bracket notation used above

print(type(mortality_data.query('Year == 1900')['DeathRate'].head()))

# 39. Display data type of double bracket notation used above

print(type(mortality_data.query('Year == 1900')[['DeathRate']].head()))

# 40. Using double bracket notation, access data using query() function (Year==1900), for AgeGroup and DeathRate columns, display first 5 records
# Note: *must* use brackets with more than one column (returns DataFrame)

print(mortality_data.query('Year == 1900')[['AgeGroup', 'DeathRate']].head())

# Access rows with loc[] accessor

# Access rows with loc[] accessor (by labels) using list or slice, here using list
# 41. Dislpay row values for 0, 5, and 10 index 'labels' (Note: loc[] interpreted as 'label' of index--*not* integer position!)
# Note: here, they are the 1st, 6th, and 11th rows

mortality_data.loc[[0,5,10]]

# 42. verify index 'labels' and row values
mortality_data.head(11)

# 43. Access rows using *slice* (loc[] accessor includes stop value)
# generic syntax: loc[start:stop:step]

mortality_data.loc[4:6]

# 44. Access rows using *slice*, include start, stop, and step values)

mortality_data.loc[0:20:5]

# 45. Access rows using conditional expression (Year=1917)

mortality_data.loc[mortality_data.Year == 1917]

# Access rows and columns with loc[] accessor 

# 46. Display first and last 5 rows, of Year and AgeGroup columns using loc[] accessor, with slice and list)
# Note: Here, includes all rows. However, will only display first and last 5 records

mortality_data.loc[:,['Year','AgeGroup']]

# Access subset of rows and columns with loc[] accessor

# 47. Using loc[] accessor with lists for row and column labels, display row values for 0, 5, and 10 index 'labels', only for AgeGroup and DeathRate columns)

mortality_data.loc[[0,5,10],['AgeGroup','DeathRate']]

# 48. Here, using slices of row and column labels

mortality_data.loc[4:6,'AgeGroup':'DeathRate']

# Access rows with iloc[] accessor 

# access rows with iloc[] accessor (by positions) using list or slice, here using list
# 49. Display first, sixth, and eleventh rows (Note: iloc[] interpreted as 'integer' position of index-- not label!)

mortality_data.iloc[[9,5,10]]

# 50. verify index 'positions' and row values

mortality_data.head(11)

# 51. Access rows using *slice* (iloc[] accessor does *not* include stop value!)
# generic syntax: iloc[start:stop:step]

mortality_data.iloc[4:6]

# 52. Access rows using *slice*, include start, stop, and step values)

mortality_data.iloc[0:20:5]

# Access subset of rows and columns with iloc[] accessor-- can use negative values

# 53. Display rows and columns using index positions and lists

mortality_data.iloc[[0,5,10],[1,2]]

# 54. Display rows and columns using index positions and slices

mortality_data.iloc[4:7,1:3]

# 55. Use iloc[] accessor with negative value

mortality_data.iloc[-10:]
