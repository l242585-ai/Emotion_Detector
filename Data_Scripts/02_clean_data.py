"""
Data Cleaning Script for Emotion Detection Project
Member 2 Task - Deliverable 1, Phase 2

This script performs the following cleaning operations:
1. Checks for missing values
2. Removes duplicate entries
3. Handles text length outliers using IQR method
4. Verifies no train/test leakage
5. Saves cleaned data for further processing

Author: Member 2
Date: October 2026
"""

import pandas as pd
import numpy as np
import os


def check_missing_values(df, split_name):
    """Check and report missing values in the dataset"""
    print(f"\n{'='*50}")
    print(f"Missing Value Check - {split_name}")
    print('='*50)
    
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Percentage': missing_percentages.values
    })
    
    print(missing_df)
    
    if missing_counts.sum() == 0:
        print(f"✓ No missing values found in {split_name}")
    else:
        print(f"⚠ Found {missing_counts.sum()} missing values in {split_name}")
    
    return missing_counts.sum()


def remove_duplicates(df, split_name):
    """Remove duplicate entries based on text content"""
    print(f"\n{'='*50}")
    print(f"Duplicate Removal - {split_name}")
    print('='*50)
    
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=['text'], keep='first')
    final_count = len(df_clean)
    removed = initial_count - final_count
    
    print(f"Initial rows: {initial_count}")
    print(f"Final rows: {final_count}")
    print(f"Duplicates removed: {removed} ({(removed/initial_count)*100:.2f}%)")
    
    if removed == 0:
        print(f"✓ No duplicates found in {split_name}")
    else:
        print(f"⚠ Removed {removed} duplicates from {split_name}")
    
    return df_clean


def handle_text_length_outliers(df, split_name, lower_percentile=1, upper_percentile=99):
    """
    Handle text length outliers using IQR method
    Removes extremely short or long texts that may be noise
    """
    print(f"\n{'='*50}")
    print(f"Text Length Outlier Detection - {split_name}")
    print('='*50)
    
    # Calculate text lengths
    df['text_length'] = df['text'].str.split().str.len()
    
    # Statistics
    print(f"\nText Length Statistics:")
    print(f"Mean: {df['text_length'].mean():.2f} words")
    print(f"Median: {df['text_length'].median():.2f} words")
    print(f"Min: {df['text_length'].min()} words")
    print(f"Max: {df['text_length'].max()} words")
    print(f"Std Dev: {df['text_length'].std():.2f} words")
    
    # IQR method
    Q1 = df['text_length'].quantile(0.25)
    Q3 = df['text_length'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nIQR Analysis:")
    print(f"Q1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f} words")
    print(f"Upper bound: {upper_bound:.2f} words")
    
    # Filter outliers
    initial_count = len(df)
    df_clean = df[(df['text_length'] >= lower_bound) & (df['text_length'] <= upper_bound)].copy()
    final_count = len(df_clean)
    removed = initial_count - final_count
    
    print(f"\nOutlier Removal:")
    print(f"Initial rows: {initial_count}")
    print(f"Final rows: {final_count}")
    print(f"Outliers removed: {removed} ({(removed/initial_count)*100:.2f}%)")
    
    # Remove temporary column
    df_clean = df_clean.drop('text_length', axis=1)
    
    if removed == 0:
        print(f"✓ No outliers detected in {split_name}")
    else:
        print(f"⚠ Removed {removed} outliers from {split_name}")
    
    return df_clean


def check_train_test_leakage(train_df, val_df, test_df):
    """
    Verify no data leakage between train, validation, and test sets
    Checks if any text appears in multiple splits
    """
    print(f"\n{'='*50}")
    print("Train/Validation/Test Leakage Check")
    print('='*50)
    
    train_texts = set(train_df['text'].values)
    val_texts = set(val_df['text'].values)
    test_texts = set(test_df['text'].values)
    
    # Check overlaps
    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)
    
    print(f"\nDataset Sizes:")
    print(f"Train: {len(train_texts)} unique texts")
    print(f"Validation: {len(val_texts)} unique texts")
    print(f"Test: {len(test_texts)} unique texts")
    
    print(f"\nLeakage Analysis:")
    print(f"Train-Validation overlap: {len(train_val_overlap)} texts")
    print(f"Train-Test overlap: {len(train_test_overlap)} texts")
    print(f"Validation-Test overlap: {len(val_test_overlap)} texts")
    
    total_leakage = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    
    if total_leakage == 0:
        print(f"\n✓ No data leakage detected - all splits are clean!")
    else:
        print(f"\n⚠ WARNING: Found {total_leakage} instances of data leakage!")
        if len(train_val_overlap) > 0:
            print(f"  Example train-val overlap: {list(train_val_overlap)[:3]}")
        if len(train_test_overlap) > 0:
            print(f"  Example train-test overlap: {list(train_test_overlap)[:3]}")
    
    return total_leakage == 0


def generate_cleaning_report(train_stats, val_stats, test_stats):
    """Generate a summary report of the cleaning process"""
    print(f"\n{'='*60}")
    print("DATA CLEANING SUMMARY REPORT")
    print('='*60)
    
    print("\nCleaning Statistics:")
    print(f"{'Split':<15} {'Initial':<10} {'Final':<10} {'Removed':<10} {'% Removed':<10}")
    print('-'*60)
    
    for split, stats in [('Train', train_stats), ('Validation', val_stats), ('Test', test_stats)]:
        initial = stats['initial']
        final = stats['final']
        removed = initial - final
        pct = (removed/initial)*100 if initial > 0 else 0
        print(f"{split:<15} {initial:<10} {final:<10} {removed:<10} {pct:<10.2f}%")
    
    total_initial = sum(s['initial'] for s in [train_stats, val_stats, test_stats])
    total_final = sum(s['final'] for s in [train_stats, val_stats, test_stats])
    total_removed = total_initial - total_final
    total_pct = (total_removed/total_initial)*100 if total_initial > 0 else 0
    
    print('-'*60)
    print(f"{'TOTAL':<15} {total_initial:<10} {total_final:<10} {total_removed:<10} {total_pct:<10.2f}%")
    
    print("\n✓ Data cleaning completed successfully!")
    print(f"✓ Cleaned data saved to data/processed/")


def main():
    """Main execution function"""
    print("="*60)
    print("EMOTION DETECTION - DATA CLEANING PIPELINE")
    print("Member 2 Task - Deliverable 1")
    print("="*60)
    
    # Check if raw data exists
    if not os.path.exists('data/raw/train.csv'):
        print("\n❌ ERROR: Raw data not found!")
        print("Please run '01_download_data.py' first to download the dataset.")
        return
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Load raw data
    print("\n📂 Loading raw data...")
    train_df = pd.read_csv('data/raw/train.csv')
    val_df = pd.read_csv('data/raw/validation.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    
    print(f"✓ Loaded Train: {len(train_df)} rows")
    print(f"✓ Loaded Validation: {len(val_df)} rows")
    print(f"✓ Loaded Test: {len(test_df)} rows")
    
    # Store initial counts
    stats = {
        'train': {'initial': len(train_df)},
        'val': {'initial': len(val_df)},
        'test': {'initial': len(test_df)}
    }
    
    # Step 1: Check missing values
    print("\n" + "="*60)
    print("STEP 1: MISSING VALUE CHECK")
    print("="*60)
    check_missing_values(train_df, "Train")
    check_missing_values(val_df, "Validation")
    check_missing_values(test_df, "Test")
    
    # Step 2: Remove duplicates
    print("\n" + "="*60)
    print("STEP 2: DUPLICATE REMOVAL")
    print("="*60)
    train_df = remove_duplicates(train_df, "Train")
    val_df = remove_duplicates(val_df, "Validation")
    test_df = remove_duplicates(test_df, "Test")
    
    # Step 3: Handle outliers
    print("\n" + "="*60)
    print("STEP 3: TEXT LENGTH OUTLIER HANDLING")
    print("="*60)
    train_df = handle_text_length_outliers(train_df, "Train")
    val_df = handle_text_length_outliers(val_df, "Validation")
    test_df = handle_text_length_outliers(test_df, "Test")
    
    # Step 4: Check for leakage
    print("\n" + "="*60)
    print("STEP 4: DATA LEAKAGE VERIFICATION")
    print("="*60)
    leakage_free = check_train_test_leakage(train_df, val_df, test_df)
    
    # Store final counts
    stats['train']['final'] = len(train_df)
    stats['val']['final'] = len(val_df)
    stats['test']['final'] = len(test_df)
    
    # Step 5: Save cleaned data
    print("\n" + "="*60)
    print("STEP 5: SAVING CLEANED DATA")
    print("="*60)
    
    train_df.to_csv('data/processed/train_cleaned.csv', index=False)
    val_df.to_csv('data/processed/val_cleaned.csv', index=False)
    test_df.to_csv('data/processed/test_cleaned.csv', index=False)
    
    print(f"✓ Saved: data/processed/train_cleaned.csv ({len(train_df)} rows)")
    print(f"✓ Saved: data/processed/val_cleaned.csv ({len(val_df)} rows)")
    print(f"✓ Saved: data/processed/test_cleaned.csv ({len(test_df)} rows)")
    
    # Generate final report
    generate_cleaning_report(stats['train'], stats['val'], stats['test'])
    
    print("\n" + "="*60)
    print("✓ DATA CLEANING PIPELINE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
