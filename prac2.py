#Read data from CSV and JSON file into data frame

import pandas as pd

# Read CSV file into data frame
csvDataFrame = pd.read_csv("carsdata.csv")
print("Data frame from CSV file:\n", csvDataFrame)

#Read JSON file into data frame
# jsoncsvDataFrame = pd.read_json("demo.json")
# print("Data frame from JSON file:\n", jsoncsvDataFrame)



# Handle null values

# For analyzing number of rows and columns
print("\nShape of the data frame:", csvDataFrame.shape)

# Display null values
print("\nNull values in the data frame:\n", csvDataFrame.isnull())

# Number of null values
print("\nNumber of null values in each column:\n", csvDataFrame.isnull().sum())

# Filling null values
filled_csvDataFrame = csvDataFrame.fillna(value=0)
print("\nData frame after filling null values with 0:\n", filled_csvDataFrame)

# Filling null values with previous values
filled_csvDataFrame = csvDataFrame.fillna(method="pad")
print("\nData frame after filling null values with previous values:\n", filled_csvDataFrame)

# Filling null values with next values
filled_csvDataFrame = csvDataFrame.fillna(method="bfill")
print("\nData frame after filling null values with next values:\n", filled_csvDataFrame)

# Filling null values with previous values of column
filled_csvDataFrame = csvDataFrame.fillna(method="pad", axis=1)
print("\nData frame after filling null values with previous values of column:\n", filled_csvDataFrame)

# Filling null values with next values of next column
filled_csvDataFrame = csvDataFrame.fillna(method="bfill", axis=1)
print("\nData frame after filling null values with next values of next column:\n", filled_csvDataFrame)



# Filtering

filtered_data = csvDataFrame.loc[csvDataFrame["price"] < 9000000]
print("\nFiltered data where amount is less than 500:\n", filtered_data)

filtered_data = csvDataFrame.loc[(csvDataFrame["price"] < 9000000) & (csvDataFrame["amount"] < 500)]
print("\nFiltered data where amount is less than 500 and amount is less than 500:\n", filtered_data)

filtered_data = csvDataFrame.loc[csvDataFrame["price"].str.contains("ist")]
print("\nFiltered data where customer column contains 'ist':\n", filtered_data)

filtered_data = csvDataFrame.loc[~csvDataFrame["price"].str.contains("ist")]
print("\nFiltered data where customer column does not contain 'ist':\n", filtered_data)

filtered_data = csvDataFrame.loc[csvDataFrame["price"].str.startswith("e")]
print("\nFiltered data where customer column starts with 'e':\n", filtered_data)

filtered_data = csvDataFrame.loc[csvDataFrame["price"].str.endswith("w")]
print("\nFiltered data where customer column ends with 'w':\n", filtered_data)

# Sorting
csvDataFrame.sort_values(by='price', ascending=False, inplace=True)
print("\nData frame after sorting by amount in descending order:\n", csvDataFrame)

# Sorting by multiple columns
csvDataFrame.sort_values(by=['cars', 'price'], ascending=[True, False], inplace=True)
print("\nData frame after sorting by customer (ascending) and amount (descending):\n", csvDataFrame)


# Sorting with null values at the end
csvDataFrame.sort_values(by='price', ascending=False, na_position='last', inplace=True)
print("\nData frame after sorting by amount in descending order with null values at the end:\n", csvDataFrame)