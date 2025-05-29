import pandas as pd



def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() # to avoid warnings

    # 1. remove duplicates
    df = df.drop_duplicates()

    # 2. Moved to the end

    # 3. Missing AC -> Fill median
    df['AC'] = df['AC'].fillna(df['AC'].median())

    # 4. Missing FM -> FIll median
    df['FM'] = df['FM'].fillna(df['FM'].median())

    # 5. Missing UC ->  FIll median

    df['UC'] = df['UC'].fillna(df['UC'].median())

    # 6. Missing DL -> fill Median for each class

    df['DL'] = df.groupby('CLASS')['DL'].transform(lambda x: x.fillna(x.median()))

    # 7. Missing DS -> fill with then most common

    df['DS'] = df['DS'].fillna(df['DS'].mode().iloc[0])

    # 8. Missing DP -> fill with the median

    df['DP'] = df['DP'].fillna(df['DP'].median())

    # 9. Missing ASTV -> with the median for each class

    df['ASTV'] = df.groupby('CLASS')['ASTV'].transform(lambda x: x.fillna(x.median()))

    # 10. Missing MSTV -> FIll median for each class

    df['MSTV'] = df.groupby('CLASS')['MSTV'].transform(lambda x: x.fillna(x.median()))

    # 11. Missing ALTV -> fill  with median for each class

    df['ALTV'] = df.groupby('CLASS')['ALTV'].transform(lambda x: x.fillna(x.median()))

    # 12. Missing MLTV -> fill  with median for each class

    df['MLTV'] = df.groupby('CLASS')['MLTV'].transform(lambda x: x.fillna(x.median()))

    # 13. Moved to the end.

    # 14. Missing Min -> fill with median for each class

    df['Min'] = df.groupby('CLASS')['Min'].transform(lambda x: x.fillna(x.median()))


    # 15. Missing Max -> fill with median for each class

    df['Max'] = df.groupby('CLASS')['Max'].transform(lambda x: x.fillna(x.median()))

    # 16. Missing Nmax -> fill with rounded avg for each class

    df['Nmax'] = df.groupby('CLASS')['Nmax'].transform(lambda x: x.fillna(round(x.mean())))

    df['Nmax'] = df['Nmax'].astype(int)

    # 17. Missing Nzeros -> fill with median for each class


    df['Nzeros'] = df.groupby('CLASS')['Nzeros'].transform(lambda x: x.fillna(x.median()))

    # 18. Missing Mode -> fill with median for each class>

    df['Mode'] = df.groupby('CLASS')['Mode'].transform(lambda x: x.fillna(x.median()))

    # 19. Missing Mean -> fill with value from Median (the column), then median each class

    mask_mean = df['Mean'].isna() & df['Median'].notna()
    df.loc[mask_mean, 'Mean'] = df.loc[mask_mean, 'Median']

    df['Mean'] = df.groupby('CLASS')['Mean'].transform(lambda x: x.fillna(x.median()))


    # 20. Missing Median -> fill with value from Mean (the column), then median each class

    mask = df['Median'].isna() & df['Mean'].notna()
    df.loc[mask, 'Median'] = df.loc[mask, 'Mean']

    df['Median'] = df.groupby('CLASS')['Median'].transform(lambda x: x.fillna(x.median()))

    # 21. Missing Variance -> fill Median for each class

    df['Variance'] = df.groupby('CLASS')['Variance'].transform(lambda x: x.fillna(x.median()))


    # 22. Missing Tendency -> fill with most common for each class

    df['Tendency'] = df.groupby('CLASS')['Tendency'].transform(lambda x: x.fillna(x.median()))


    # 2. Missing LB -> value from Median (the column), then value from Mean (the column) - strong positive correlation
    mask_median = df['LB'].isna() & df['Median'].notna()
    df.loc[mask_median, 'LB'] = df.loc[mask_median, 'Median']

    mask_mean = df['LB'].isna() & df['Mean'].notna()
    df.loc[mask_mean, 'LB'] = df.loc[mask_mean, 'Mean']


    # 13. Missing Width - > fill with difference max - min

    df['Width'] = df['Width'].fillna(df['Max'] - df['Min'])

    return df









if __name__ == "__main__":
    df = pd.read_csv("cardiotocography_v2.csv", encoding="utf-8")
    filled_df = fill_missing_values(df)
    print(filled_df.isnull().sum())
    filled_df.to_csv("cardiotocography_filled.csv", index=False, encoding="utf-8")

