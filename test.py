import pandas as pd
import numpy as np
import ssl

from pandas.core.arrays import string_
from pandas.core.frame import DataFrame

# This function is used to replace missing values in columns that should have a binary value
def replace_missing_values(df : pd.DataFrame, col: str, v1: str, v2: str):
  print(f"\nINFO: replace_mising_values({type(df)}, {type(col)})\n")

  a = df[col].apply(pd.value_counts)
  print(a)

def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    url = 'https://raw.githubusercontent.com/bryonbaker/datasets/main/SIT720/Ass1/hypothyroid.csv'
    fullht_df = pd.read_csv(url)

    print(fullht_df.head(n=100))

    # Get the first 500 rows from the dataset and use that for the rest of the assignment.
    ht_df = fullht_df.head(n=500)

    # Display the dataset's dimension
    print(f"Working dataset dimension is: {ht_df.shape}\n")

    # Get the first 500 rows from the dataset and use that for the rest of the assignment.
    ht_df = fullht_df.head(n=500)

    # Cells with missing data have a '?' in them. 
    # First replace ? with np.NaN so we can utilise some other nice Pandas dataframe methods. We can use a global replace because, upon dataset ins[ection, the unknown ('?') only exists in the numeric columns.
    # Convert the value columns from text to numeric.
    # Calculate the median value for the numeric-data coluimns
    # Replace the NaN values with a reasonable value. For this exercise we have chosen the mean for the column
    # Recalculate the median value for the numeric-data coluimns

    # Prepare the data so it is calculable
    ht_df = ht_df.replace('?', np.NaN)                                                        # Replace with NaN so many of the Pandas functions will work.
    ht_df[["TSH","T3","TT4","FTI"]] = ht_df[["TSH","T3","TT4","FTI"]].apply(pd.to_numeric)    # CSV loads as text. Convert the cells to numeric
    print("\nCore Numerical Data Columns:\n")
    print(ht_df[["TSH","T3","TT4","FTI"]])

    # Calculate the Mean and Median prior to replacing missing values
    mean = ht_df[["TSH","T3","TT4","FTI"]].mean(skipna=True)
    median = ht_df[["TSH","T3","TT4","FTI"]].median(skipna=True)
    print("=======================\n")
    print("\nPre Data Modification:")
    print("\n======================")
    print(f"Mean of each column is:\n{mean}\n")
    print(f"The Median of each column is:\n{median}")

    # Replace the NaN's of the numeric columns with the mean
    ht_df["TSH"] = ht_df["TSH"].fillna(mean["TSH"])
    ht_df["T3"] = ht_df["TSH"].fillna(mean["T3"])
    ht_df["TT4"] = ht_df["TSH"].fillna(mean["TT4"])
    ht_df["FTI"] = ht_df["TSH"].fillna(mean["FTI"])

    # Replace the M/F missing values with the most frequently occuring gender provided "pregnant" is false. Otherwise set the value to F.
    # replace_missing_values(df=ht_df, col="sex", v1="M", v2="F")
    tmp_col = "sex-predict"
    ht_df[tmp_col] = ht_df["sex"]
    for (index, row_series) in ht_df.iterrows():
        if pd.isna(row_series["sex"]):
            print(f"DEBUG: Found NaN at: {index}")
            print(f"DEBUG: {ht_df.iloc[index]}")
            ht_df.at[index, tmp_col] = calc_gender(ht_df.at[index, 'pregnant'])
            print(f"DEBUG: {type(ht_df.iloc[index])}")
    
    # Copy over any NaN values in the sex column using the value from the temporary column
    ht_df["sex"] = ht_df["sex"].fillna(ht_df[tmp_col])
    ht_df = ht_df.drop([tmp_col], axis=1)       # Drop the temporary column

    print(ht_df.head(n=100))


    #print(f"The number of missing values in Sex is: {ht_df["sex"].isna().sum()}\n")
    # 

    mean_post = ht_df[["TSH","T3","TT4","FTI"]].mean(skipna=True)
    median_post = ht_df[["TSH","T3","TT4","FTI"]].median(skipna=True)
    print("=======================\n")
    print("Post Data Modification:\n")
    print("=======================\n")
    print(f"Mean of each column is:\n{mean_post}\nThe Median is:\n{median_post}")

# This function calculates a value for the missing gender feature. 
# If the person is pregnant it must be female. If not pregnant then it defaults to male.
# REVISIT: Update the non-pregnant case after you get an answer from the lecturer.
def calc_gender(pregnant: str):
    if pregnant == "t":
        gender = "F"
    else:
        gender = "M"
    return gender

if __name__ == "__main__":
    main()