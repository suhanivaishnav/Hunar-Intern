import pandas as pd
df=pd.read_csv(r'C:/Users/pc/Downloads/New folder/food_coded.csv')
print(df)
df.isnull()
df.duplicated(subset=None, keep=False)
df.drop(columns=df.columns[df.T.duplicated()], axis=1)
df=df.fillna(value=df['calories_chicken'].mean())
df=df.fillna(value=df['calories_day'].mode())
for col in df.columns:
    if df[col].dtype == "object":  # Categorical columns
        df[col].fillna(df[col].mode()[0])
    else:  # Numerical columns
        if df[col].skew() > 1:  # If the column is highly skewed
            df[col].fillna(df[col].median())
        else:
            df[col].fillna(df[col].mean())
