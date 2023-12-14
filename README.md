# Pandas Operations and Code Snippets

## Adding/Removing Columns and Rows

| **Operation**               | **Pandas Code**                                    | **Description**                                        |
|-----------------------------|-----------------------------------------------------|--------------------------------------------------------|
| Add a New Column            | `df['Gender'] = ['Female', 'Male', 'Male']`         | Adds a new column named 'Gender' with specified values.  |
| Add a New Row               | `df.loc[len(df.index)] = [20,'Category','2023-01-03','m']` | Adds a new row using `loc` and the index of the last row. |
| Remove a Column             | `df = df.drop('Age', axis=1)`                      | Removes the 'Age' column from the DataFrame.             |
| Rename Columns              | `df = df.rename(columns={'Name': 'Full Name'})`     | Renames the 'Name' column to 'Full Name'.                |

## DataFrame Manipulation

| **Operation**                               | **Pandas Code**                                           | **Description**                                                                                            |
|----------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Filter Rows Based on a Condition             | `df_filtered = df[df['Age'] > 30]`                       | Creates a new DataFrame with rows where age is greater than 30.                                              |
| Sort Data by Column                         | `df_sorted = df.sort_values(by='Age')`                  | Sorts the DataFrame by the 'Age' column in ascending order.                                                   |
| Fill Missing Values                         | `df['Age'] = df['Age'].fillna(df['Age'].mean())`         | Fills missing values in the 'Age' column with the mean.                                                       |
| Drop Rows with Missing Values               | `df_cleaned = df.dropna()`                               | Removes rows with missing values from the DataFrame.                                                          |
| Aggregate Data                              | `df_grouped = df.groupby('Gender')['Age'].mean()`       | Groups data by 'Gender' and calculates the mean age for each group.                                          |
| Merge DataFrames                            | `df_merged = pd.merge(df, df2, on='Name')`               | Merges two DataFrames based on the common 'Name' column, combining information from both DataFrames.        |
| Selecting Columns                           | `selected_columns = df[['Column1', 'Column2']]`           | Select specific columns from a DataFrame.                                                                 |
| Filtering Rows with Multiple Conditions    | `filtered_data = df[(df['Column1'] > 30) & (df['Column2'] == 'Category')]` | Filter rows based on multiple conditions using logical operators.                                         |
| Creating a New Column Based on Conditions   | ``` conditions = [df['Column1'] > 30, df['Column1'] <= 30] choices = ['High', 'Low'] df['Category'] = np.select(conditions, choices, default='Medium') ``` | Create a new column based on conditions using `numpy` and `np.select`.                                    |
| Grouping and Aggregating Data              | `grouped_data = df.groupby('Category')['Value'].agg(['mean', 'sum', 'count'])` | Group data by a column and calculate aggregate functions.                                                |
| Pivoting Data                              | `pivoted_data = df.pivot_table(index='Date', columns='Category', values='Value', aggfunc='sum')` | Pivot a DataFrame to reshape it.                                                                          |
| Merging DataFrames on Multiple Columns    | `merged_data = pd.merge(df1, df2, on=['Column1', 'Column2'])` | Merge DataFrames based on multiple columns.                                                              |
| Handling Dates                             | ```python df['Date'] = pd.to_datetime(df['Date']) df['Year'] = df['Date'].dt.year ``` | Convert string to datetime and extract components.                                                       |
| Resampling Time Series Data                | `df_resampled = df.set_index('Date').resample('M').mean()` | Resample time series data to a different frequency.                                                      |
| Dropping Duplicates                        | `df_no_duplicates = df.drop_duplicates(subset=['Column1', 'Column2'])` | Remove duplicate rows based on specific columns.                                                        |
| Applying a Function to Columns             | `df['Squared'] = df['Value'].apply(lambda x: x**2)`       | Use the `apply` function to apply a custom function to a column.                                          |
| Handling Text Data                         | `df['First_Name'] = df['Full_Name'].str.split(' ').str[0]` | Extract information from text data.                                                                      |
| Creating Dummy Variables (One-Hot Encoding) | `df_encoded = pd.get_dummies(df, columns=['Category'], prefix='Category', drop_first=True)` | Convert categorical variables into dummy variables (one-hot encoding).                                   |
| Applying Functions Element-Wise            | ```python def double_value(x): return x * 2 df['Doubled_Value'] = df['Value'].apply(double_value) ``` | Apply a function element-wise to a column.                                                               |

This Markdown file provides an overview of various Pandas operations and code snippets for DataFrame manipulation.
