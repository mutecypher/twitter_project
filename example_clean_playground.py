import pandas as pd

# Example dataframe with mixed string and float values
df = pd.DataFrame({
    'my_column': [1.0, '2', '3.14', 'invalid', '5.5']
})

# Convert the column to floats, filtering out non-numeric values
df['my_column'] = df['my_column'].apply(lambda x: float(x) if isinstance(x, float) or isinstance(
    x, int) or (isinstance(x, str) and x.replace('.', '').isdigit()) else None).dropna()

df = df.reset_index(drop=True)
df = df.dropna()

print(df)
