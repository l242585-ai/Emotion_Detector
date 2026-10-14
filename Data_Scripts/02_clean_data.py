import pandas as pd
import os

# Load data
print("Loading data...")
train = pd.read_csv('data/raw/train.csv')
val = pd.read_csv('data/raw/validation.csv')
test = pd.read_csv('data/raw/test.csv')

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# 1. Check missing values
print("\n1. Checking missing values...")
print("Train missing:", train.isnull().sum().sum())
print("Val missing:", val.isnull().sum().sum())
print("Test missing:", test.isnull().sum().sum())

# 2. Remove duplicates
print("\n2. Removing duplicates...")
train_before = len(train)
val_before = len(val)
test_before = len(test)

train = train.drop_duplicates(subset=['text'])
val = val.drop_duplicates(subset=['text'])
test = test.drop_duplicates(subset=['text'])

print(f"Train: {train_before} -> {len(train)} (removed {train_before - len(train)})")
print(f"Val: {val_before} -> {len(val)} (removed {val_before - len(val)})")
print(f"Test: {test_before} -> {len(test)} (removed {test_before - len(test)})")

# 3. Handle text length outliers (IQR method)
print("\n3. Handling text length outliers (IQR)...")

def remove_outliers(df, name):
    df['text_length'] = df['text'].str.split().str.len()
    
    Q1 = df['text_length'].quantile(0.25)
    Q3 = df['text_length'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    before = len(df)
    df = df[(df['text_length'] >= lower) & (df['text_length'] <= upper)]
    after = len(df)
    
    print(f"{name}: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}")
    print(f"  Range: [{lower:.1f}, {upper:.1f}], removed {before - after}")
    
    df = df.drop('text_length', axis=1)
    return df

train = remove_outliers(train, "Train")
val = remove_outliers(val, "Val")
test = remove_outliers(test, "Test")

# 4. Check train/test leakage
print("\n4. Checking train/test leakage...")
train_texts = set(train['text'])
val_texts = set(val['text'])
test_texts = set(test['text'])

train_val_overlap = len(train_texts & val_texts)
train_test_overlap = len(train_texts & test_texts)
val_test_overlap = len(val_texts & test_texts)

print(f"Train-Val overlap: {train_val_overlap}")
print(f"Train-Test overlap: {train_test_overlap}")
print(f"Val-Test overlap: {val_test_overlap}")

if train_val_overlap + train_test_overlap + val_test_overlap == 0:
    print("No leakage!")
else:
    print("Leakage detected!")

# 5. Save cleaned data
print("\n5. Saving cleaned data...")
os.makedirs('data/processed', exist_ok=True)

train.to_csv('data/processed/train_cleaned.csv', index=False)
val.to_csv('data/processed/val_cleaned.csv', index=False)
test.to_csv('data/processed/test_cleaned.csv', index=False)

print(f"Saved train: {len(train)} rows")
print(f"Saved val: {len(val)} rows")
print(f"Saved test: {len(test)} rows")
print("\nCleaning complete!")
