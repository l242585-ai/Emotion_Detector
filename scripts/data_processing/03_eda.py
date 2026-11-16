"""
03_eda.py - Exploratory Data Analysis for Emotion Detection Dataset
Member 3: EDA & Feature Engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import seaborn as sns
from collections import Counter
import os
import re

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)

EMOTION_LABELS = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

# ── Load cleaned data ──────────────────────────────────────────────────────────
print("Loading cleaned data...")
train = pd.read_csv('data/processed/train_cleaned.csv')
val   = pd.read_csv('data/processed/val_cleaned.csv')
test  = pd.read_csv('data/processed/test_cleaned.csv')

# Map numeric labels to names if needed
for df in [train, val, test]:
    if df['label'].dtype in [int, np.int64]:
        df['emotion'] = df['label'].map(EMOTION_LABELS)
    else:
        df['emotion'] = df['label']

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# ── 1. Class Distribution Bar Chart ───────────────────────────────────────────
print("\n1. Plotting class distribution...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle('Class Distribution Across Splits', fontsize=14, fontweight='bold')

for ax, (df, split_name) in zip(axes, [(train, 'Train'), (val, 'Validation'), (test, 'Test')]):
    counts = df['emotion'].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)], edgecolor='black', linewidth=0.5)
    ax.set_title(f'{split_name} (n={len(df):,})', fontsize=12)
    ax.set_xlabel('Emotion', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.tick_params(axis='x', rotation=30)
    for bar, val_count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val_count), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figures/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/01_class_distribution.png")

# ── 2. Text Length Distribution Histogram ─────────────────────────────────────
print("\n2. Plotting text length distributions...")

train['text_len'] = train['text'].str.split().str.len()

fig, ax = plt.subplots(figsize=(10, 5))
emotions = sorted(train['emotion'].unique())
for emotion, color in zip(emotions, PALETTE):
    subset = train[train['emotion'] == emotion]['text_len']
    ax.hist(subset, bins=30, alpha=0.55, label=emotion, color=color, edgecolor='none')

ax.set_title('Text Length Distribution by Emotion (Train)', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Words', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.legend(title='Emotion', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/02_text_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/02_text_length_distribution.png")

# ── 3. Word Clouds Per Emotion ─────────────────────────────────────────────────
print("\n3. Generating word clouds per emotion...")

try:
    from wordcloud import WordCloud

    STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'so', 'just', 'feel', 'feeling', 'felt', 'im', 'ive',
        'dont', 'cant', 'it', 'like', 'really', 'get', 'got', 'know', 'think', 'one',
        'also', 'would', 'could', 'still', 'even', 'when', 'very', 'not', 'no', 'now',
    }

    n_emotions = len(emotions)
    cols = 3
    rows = (n_emotions + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    fig.suptitle('Word Clouds by Emotion', fontsize=15, fontweight='bold')

    for idx, emotion in enumerate(emotions):
        text_blob = ' '.join(train[train['emotion'] == emotion]['text'].tolist())
        # Basic cleaning
        text_blob = re.sub(r'[^a-zA-Z\s]', '', text_blob.lower())
        words = [w for w in text_blob.split() if w not in STOPWORDS and len(w) > 2]
        clean_text = ' '.join(words)

        wc = WordCloud(
            width=400, height=300,
            background_color='white',
            max_words=80,
            colormap='viridis',
            collocations=False
        ).generate(clean_text)

        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(emotion.capitalize(), fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    # Hide any extra subplots
    for idx in range(n_emotions, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/03_word_clouds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: figures/03_word_clouds.png")

except ImportError:
    print("   wordcloud not installed — skipping word clouds. Run: pip install wordcloud")

# ── 4. Top Words Per Emotion ───────────────────────────────────────────────────
print("\n4. Plotting top words per emotion...")

STOPWORDS_SET = {
    'i', 'me', 'my', 'we', 'you', 'your', 'he', 'she', 'it', 'its', 'they', 'them',
    'the', 'a', 'an', 'and', 'but', 'or', 'of', 'at', 'by', 'for', 'with', 'to',
    'in', 'on', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
    'do', 'did', 'so', 'just', 'feel', 'feeling', 'felt', 'im', 'dont', 'it',
    'that', 'this', 'as', 'up', 'get', 'got', 'like', 'not', 'if', 'when', 'from',
    'really', 'also', 'would', 'could', 'very', 'no', 'now', 'one', 'know', 'about',
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
fig.suptitle('Top 10 Words Per Emotion (Train)', fontsize=14, fontweight='bold')

for idx, emotion in enumerate(emotions):
    texts = train[train['emotion'] == emotion]['text'].tolist()
    all_words = []
    for sentence in texts:
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence.lower())
        all_words.extend([w for w in sentence.split() if w not in STOPWORDS_SET and len(w) > 2])

    top_10 = Counter(all_words).most_common(10)
    words_list = [item[0] for item in top_10]
    counts_list = [item[1] for item in top_10]

    axes[idx].barh(words_list[::-1], counts_list[::-1], color=PALETTE[idx], edgecolor='black', linewidth=0.4)
    axes[idx].set_title(f'{emotion.capitalize()}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Frequency', fontsize=9)
    axes[idx].tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig('figures/04_top_words_per_emotion.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/04_top_words_per_emotion.png")

# ── 5. Correlation / Co-occurrence Analysis ────────────────────────────────────
print("\n5. Plotting label distribution summary (correlation heatmap)...")

# Pivot table: emotion counts per split for comparison
emotion_counts = pd.DataFrame({
    'Train': train['emotion'].value_counts(),
    'Validation': val['emotion'].value_counts(),
    'Test': test['emotion'].value_counts(),
}).fillna(0).astype(int)

# Normalize to proportions
emotion_props = emotion_counts.div(emotion_counts.sum(axis=0), axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    emotion_props,
    annot=emotion_counts.values,
    fmt='d',
    cmap='YlOrRd',
    linewidths=0.5,
    ax=ax,
    cbar_kws={'label': 'Proportion'}
)
ax.set_title('Emotion Distribution Across Splits\n(color = proportion, numbers = count)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Split', fontsize=11)
ax.set_ylabel('Emotion', fontsize=11)

plt.tight_layout()
plt.savefig('figures/05_emotion_split_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figures/05_emotion_split_heatmap.png")

# ── Summary Statistics ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EDA SUMMARY")
print("="*55)
print(f"\nTrain text length stats:")
print(train['text_len'].describe().to_string())
print(f"\nClass counts (Train):")
print(train['emotion'].value_counts().to_string())
print(f"\nClass imbalance ratio (max/min): "
      f"{train['emotion'].value_counts().max() / train['emotion'].value_counts().min():.2f}x")
print("\n✓ EDA complete — figures saved to figures/")
