# error_analysis.py - Error analysis and bias check for emotion classifier
# Member 3: Evaluation & Error Analysis
#
# Usage:
#   python src/error_analysis.py
#
# Expects:
#   results/predictions.csv  (from Member 2)
#   data/processed/test_cleaned.csv
#
# Outputs:
#   results/error_analysis.txt
#   results/errors_by_length.png
#   results/emotion_confusion_pairs.png

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sys
sys.path.insert(0, '.')
from src.utils.metrics import EMOTION_LABELS

os.makedirs('results', exist_ok=True)

# Load data
print("Loading data...")
pred_path = 'results/predictions.csv'
test_path = 'data/processed/test_cleaned.csv'

preds = pd.read_csv(pred_path)
test  = pd.read_csv(test_path).reset_index(drop=True)

# Align lengths
min_len = min(len(preds), len(test))
preds = preds.iloc[:min_len].reset_index(drop=True)
test  = test.iloc[:min_len].reset_index(drop=True)

df = pd.DataFrame({
    'text':       test['text'],
    'true_label': preds['true_label'],
    'pred_label': preds['predicted_label'],
})
df['correct']     = df['true_label'] == df['pred_label']
df['text_length'] = df['text'].str.split().str.len()
df['true_emotion'] = df['true_label'].map(dict(enumerate(EMOTION_LABELS)))
df['pred_emotion'] = df['pred_label'].map(dict(enumerate(EMOTION_LABELS)))

errors = df[~df['correct']].copy()
print(f"Total samples : {len(df)}")
print(f"Errors        : {len(errors)} ({len(errors)/len(df)*100:.1f}%)")

lines = []  # collect output for the text file

# 1. Per-class accuracy
lines.append("=== Per-class Accuracy ===")
for i, label in enumerate(EMOTION_LABELS):
    subset = df[df['true_label'] == i]
    if len(subset) == 0:
        continue
    acc = subset['correct'].mean()
    lines.append(f"  {label:<10}: {acc:.3f}  ({subset['correct'].sum()}/{len(subset)})")

# 2. Which classes fail most
lines.append("\n=== Error Rate by Class ===")
error_rates = errors['true_emotion'].value_counts()
for emotion, count in error_rates.items():
    total = (df['true_emotion'] == emotion).sum()
    lines.append(f"  {emotion:<10}: {count} errors out of {total} ({count/total*100:.1f}%)")

# 3. Most confused emotion pairs
lines.append("\n=== Top Confusion Pairs ===")
confused = errors.groupby(['true_emotion', 'pred_emotion']).size().reset_index(name='count')
confused = confused.sort_values('count', ascending=False).head(10)
for _, row in confused.iterrows():
    lines.append(f"  {row['true_emotion']:<10} -> {row['pred_emotion']:<10}: {row['count']} times")

# 4. Performance by text length
lines.append("\n=== Accuracy by Text Length ===")
df['length_bin'] = pd.cut(df['text_length'], bins=[0, 5, 10, 20, 30, 50], labels=['1-5', '6-10', '11-20', '21-30', '31+'])
length_acc = df.groupby('length_bin', observed=True)['correct'].mean()
for bin_name, acc in length_acc.items():
    lines.append(f"  {bin_name:<8}: {acc:.3f}")

# 5. Sample misclassified examples
lines.append("\n=== Sample Misclassified Examples (10) ===")
sample = errors.sample(min(10, len(errors)), random_state=42)
for _, row in sample.iterrows():
    lines.append(f"  True: {row['true_emotion']:<10} | Pred: {row['pred_emotion']:<10} | Text: {row['text'][:80]}")

# 6. Bias check: class imbalance vs performance
lines.append("\n=== Bias Check: Class Size vs F1 ===")
class_counts = df['true_label'].value_counts().sort_index()
per_class_correct = df.groupby('true_label')['correct'].mean()
lines.append(f"  {'Emotion':<12} {'Count':>7}  {'Accuracy':>9}")
for i, label in enumerate(EMOTION_LABELS):
    if i in class_counts.index:
        lines.append(f"  {label:<12} {class_counts[i]:>7}  {per_class_correct.get(i, 0):>9.3f}")

# Print and save
full_text = '\n'.join(lines)
print('\n' + full_text)
with open('results/error_analysis.txt', 'w') as f:
    f.write(full_text)
print("\nSaved: results/error_analysis.txt")

# Plot 1: Accuracy by text length bin
fig, ax = plt.subplots(figsize=(8, 4))
length_acc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Accuracy by Text Length')
ax.set_xlabel('Word Count Range')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('results/errors_by_length.png', dpi=150)
plt.close()
print("Saved: results/errors_by_length.png")

# Plot 2: Confusion pairs heatmap (true vs predicted for errors only)
pivot = errors.pivot_table(index='true_emotion', columns='pred_emotion', aggfunc='size', fill_value=0)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='Oranges', ax=ax)
ax.set_title('Misclassification Heatmap (errors only)')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('results/emotion_confusion_pairs.png', dpi=150)
plt.close()
print("Saved: results/emotion_confusion_pairs.png")

print("\nError analysis complete.")
