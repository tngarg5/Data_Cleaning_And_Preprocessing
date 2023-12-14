# Data Cleaning Cheat Sheet

## [Part 1 -Handling Missing Values]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Identify Missing Values          | `df.isnull().sum()`                               | Get the count of missing values in each column.                      |
| Drop Rows with Missing Values    | `df.dropna()`                                     | Drop rows containing missing values.                                 |
| Drop Columns with Missing Values | `df.dropna(axis=1)`                               | Drop columns containing missing values.                              |
| Fill Missing Values with a Constant | `df.fillna(value)`                            | Fill missing values with a constant.                                 |
| Fill Missing Values with Mean/Median/Mode | `df.fillna(df.mean())`                   | Fill missing values with mean, median, or mode of the column.        |
| Forward Fill Missing Values      | `df.ffill()`                                      | Fill missing values with the previous non-null value (forward fill).|
| Backward Fill Missing Values     | `df.bfill()`                                      | Fill missing values with the next non-null value (backward fill).    |
| Interpolate Missing Values       | `df.interpolate()`                               | Interpolate missing values using various methods.                   |

## [Part 2 - Data Type Conversions]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Convert Data Type of a Column     | `df['col'] = df['col'].astype('type')`            | Convert the data type of a specific column.                          |
| Convert to Numeric               | `pd.to_numeric(df['col'], errors='coerce')`      | Convert a column to numeric, handling errors by coercing to NaN.     |
| Convert to Datetime               | `pd.to_datetime(df['col'], errors='coerce')`     | Convert a column to datetime, handling errors by coercing to NaN.    |
| Convert to Categorical            | `df['col'] = df['col'].astype('category')`        | Convert a column to categorical data type.                           |

## [Part 3 - Dealing with Duplicate]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Identify Duplicate Rows          | `df.duplicated()`                                | Identify duplicated rows in the DataFrame.                           |
| Drop Duplicate Rows              | `df.drop_duplicates()`                           | Drop rows with identical values in all columns.                      |
| Drop Duplicates in a Specific Column | `df.drop_duplicates(subset='col')`            | Drop duplicates based on a specific column.                          |
| Drop Duplicates Keeping the Last Occurrence | `df.drop_duplicates(keep='last')`         | Keep the last occurrence of duplicates and drop the rest.            |

## [Part 4 - Text Data Cleaning]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Trim Whitespace                  | `df['col'] = df['col'].str.strip()`               | Remove leading and trailing whitespaces from text.                   |
| Convert to Lowercase             | `df['col'] = df['col'].str.lower()`               | Convert text to lowercase.                                           |
| Convert to Uppercase             | `df['col'] = df['col'].str.upper()`               | Convert text to uppercase.                                           |
| Remove Specific Characters        | `df['col'] = df['col'].str.replace('[character]', '')` | Remove specific characters from text.                            |
| Replace Text Based on Pattern (Regex) | `df['col'] = df['col'].str.replace(r'[regex]', 'replacement')` | Replace text based on a regex pattern.                      |
| Split Text into Columns          | `df[['col1', 'col2']] = df['col'].str.split(',', expand=True)` | Split text in a column and expand into separate columns.     |

## [Part 5 - Categorical Data Processing]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| One-Hot Encoding                  | `pd.get_dummies(df['col'])`                      | Perform one-hot encoding on a categorical column.                    |
| Label Encoding                   | `from sklearn.preprocessing import LabelEncoder; encoder = LabelEncoder(); df['col'] = encoder.fit_transform(df['col'])` | Encode categorical labels into numerical values.           |
| Map Categories to Values          | `df['col'] = df['col'].map({'cat1': 1, 'cat2': 2})` | Map categories to specified numerical values.                 |
| Convert Category to Ordinal       | `df['col'] = df['col'].cat.codes`                | Convert categorical data to ordinal codes.                           |

## [Part 6 - Normalization and Scaling]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Min-Max Scaling                  | `from sklearn.preprocessing import MinMaxScaler; scaler = MinMaxScaler(); df['col'] = scaler.fit_transform(df[['col']])` | Scale numerical column values to a range between 0 and 1.|
| Standard Scaling (Z-Score)       | `from sklearn.preprocessing import StandardScaler; scaler = StandardScaler(); df['col'] = scaler.fit_transform(df[['col']])` | Standardize numerical column values using Z-scores.       |
| Robust Scaling (Median, IQR)     | `from sklearn.preprocessing import RobustScaler; scaler = RobustScaler(); df['col'] = scaler.fit_transform(df[['col']])` | Scale numerical column values using robust scaling.      |

## [Part 7 - Handling Outliers]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Remove Outliers with IQR         | `Q1 = df['col'].quantile(0.25); Q3 = df['col'].quantile(0.75); IQR = Q3 - Q1; df = df[~((df['col'] < (Q1 - 1.5 * IQR)) | (df['col'] > (Q3 + 1.5 * IQR)))]` | Remove outliers using the interquartile range (IQR) method.  |
| Remove Outliers with Z-Score      | `from scipy import stats; df = df[np.abs(stats.zscore(df['col'])) < 3]` | Remove outliers using the Z-score method.                  |
| Capping and Flooring Outliers    | `df['col'] = df['col'].clip(lower=lower_bound, upper=upper_bound)` | Cap or floor outliers at specified values.                |

## [Part 8 - Data Transformation]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Log Transformation               | `df['col'] = np.log(df['col'])`                   | Apply a logarithmic transformation to a column.                      |
| Square Root Transformation       | `df['col'] = np.sqrt(df['col'])`                  | Apply a square root transformation to a column.                      |
| Power Transformation (Box-Cox, Yeo-Johnson) | `from sklearn.preprocessing import PowerTransformer; pt = PowerTransformer(method='yeo-johnson'); df['col'] = pt.fit_transform(df[['col']])` | Apply power transformation using Box-Cox or Yeo-Johnson method. |
| Binning Data                     | `df['bin_col'] = pd.cut(df['col'], bins=[range])` | Bin numerical data into discrete intervals.                         |

## [Part 9 - Time Series Data Cleaning]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Set Datetime Index               | `df.set_index('datetime_col', inplace=True)`    | Set a datetime column as the DataFrame index.                        |
| Resample Time Series Data        | `df.resample('D').mean()`                        | Resample time series data to a specified frequency.                  |
| Fill Missing Time Series Data     | `df.asfreq('D', method='ffill')`                 | Fill missing values in time series data using forward fill.         |
| Time-Based Filtering              | `df['year'] = df.index.year; df[df['year'] > 2000]` | Filter time series data based on a specific time range.        |

## [Part 10 - Data Frame Operations]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Merge Data Frames                | `pd.merge(df1, df2, on='key', how='inner')`      | Merge two DataFrames based on a common key.                          |
| Concatenate Data Frames          | `pd.concat([df1, df2], axis=0)`                  | Concatenate DataFrames vertically or horizontally.                  |
| Join Data Frames                 | `df1.join(df2, on='key')`                        | Join two DataFrames based on a common key.                           |
| Pivot Table                      | `df.pivot_table(index='row', columns='col', values='value')` | Create a pivot table from a DataFrame.                      |


## [Part 11 - Column Operations]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Aggregate Functions (sum, mean, etc.) | `df.groupby('group_col').agg({'agg_col': ['sum', 'mean']})` | Perform aggregate functions on grouped data.                 |
| Rolling Window Calculations       | `df['col'].rolling(window=5).mean()`             | Calculate rolling mean using a specified window size.               |
| Expanding Window Calculations     | `df['col'].expanding().sum()`                    | Calculate expanding sum of a column.                                |

## [Part 12 - Handling Complex Data Types]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Explode List to Rows              | `df.explode('list_col')`                        | Explode a column containing lists into separate rows.               |
| Work with JSON Columns            | `df['json_col'].apply(lambda x: json.loads(x))` | Apply a function to parse JSON strings in a column.                  |
| Parse Nested Structures          | `df['new_col'] = df['struct_col'].apply(lambda x: x['nested_field'])` | Extract nested field from a column with nested structures. |

## [Part 13 **Dealing with Geospatial Data** ]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Handling Latitude and Longitude   | `df['distance'] = df.apply(lambda x: calculate_distance(x['lat'], x['long']), axis=1)` | Calculate distance between coordinates using a custom function. |
| Geocoding Addresses               | `df['coordinates'] = df['address'].apply(geocode_address)` | Perform geocoding on addresses to obtain coordinates.          |

## [Part 14 - **Data Quality Checks** ] 

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| Check for Data Consistency        | `assert df['col1'].notnull().all()`             | Assert that all values in a column are non-null.                    |
| Validate Data Ranges             | `df[(df['col'] >= low_val) & (df['col'] <= high_val)]` | Validate data ranges in a column.                             |
| Assert Data Types                | `assert df['col'].dtype == 'expected_type'`     | Assert the data type of a column.                                    |

## [Part 15 - Efficient Computations]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| **Use Vectorized Operations**    | `df['col'] = df['col1'] + df['col2']`            | Perform vectorized operations on DataFrame columns.                 |
| **Parallel Processing with Dask** | `import dask.dataframe as dd; ddf = dd.from_pandas(df, npartitions=10); result = ddf.compute()` | Utilize Dask for parallel processing of large DataFrames. |

## [Part 16 - Working with Large Datasets]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| **Sampling Data for Quick Insights** | `sampled_df = df.sample(frac=0.1)`              | Sample a fraction of the DataFrame for quick insights.              |
| **Chunking Large Files for Processing** | `for chunk in pd.read_csv('large_file.csv', chunksize=10000): process(chunk)` | Read and process large CSV files in chunks.                 |

## [Part 17 - Feature Engineering]

| **Operation**                   | **Pandas Code**                                  | **Description**                                                      |
|----------------------------------|---------------------------------------------------|----------------------------------------------------------------------|
| **Creating Polynomial Features** | `from sklearn.preprocessing import PolynomialFeatures; poly = PolynomialFeatures(degree=2); df_poly = poly.fit_transform(df[['col1', 'col2']])` | Create polynomial features from selected columns. |
| **Encoding Cyclical Features**   | `df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))` | Encode cyclical features such as hour of day using sine function. |

# [End of Data Cleaning Cheat Sheet]



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
