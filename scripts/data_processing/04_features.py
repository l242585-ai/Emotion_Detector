"""
04_features.py - Feature Engineering for Emotion Detection Dataset
Member 3: EDA & Feature Engineering

Outputs:
  - data/processed/X_train_tfidf.npz   TF-IDF sparse matrix (train)
  - data/processed/X_val_tfidf.npz     TF-IDF sparse matrix (val)
  - data/processed/X_test_tfidf.npz    TF-IDF sparse matrix (test)
  - data/processed/tfidf_vectorizer.pkl  Fitted TF-IDF vectorizer
  - data/processed/tokenized_train.pkl   Token ID sequences (train)
  - data/processed/tokenized_val.pkl     Token ID sequences (val)
  - data/processed/tokenized_test.pkl    Token ID sequences (test)
  - data/processed/label_encoder.pkl     Fitted LabelEncoder
  - data/processed/class_weights.npy     Per-class weights array
  - data/processed/feature_summary.txt   Human-readable summary
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from scipy.sparse import save_npz

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)

EMOTION_LABELS = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

# ── Load cleaned data ──────────────────────────────────────────────────────────
print("Loading cleaned data...")
train = pd.read_csv('data/processed/train_cleaned.csv')
val   = pd.read_csv('data/processed/val_cleaned.csv')
test  = pd.read_csv('data/processed/test_cleaned.csv')

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 ─ Label Encoding
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 1: Label Encoding")
print("="*55)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# If labels are already integers keep them; otherwise fit on strings
if train['label'].dtype == object:
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_val   = le.transform(val['label'])
    y_test  = le.transform(test['label'])
else:
    # Numeric labels — build encoder from emotion names for interpretability
    emotion_names = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())]
    le.fit(emotion_names)
    y_train = train['label'].values
    y_val   = val['label'].values
    y_test  = test['label'].values

print(f"Classes: {list(le.classes_) if hasattr(le, 'classes_') else EMOTION_LABELS}")
print(f"y_train shape: {y_train.shape}, unique: {np.unique(y_train)}")

# Save label encoder
with open('data/processed/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Saved: data/processed/label_encoder.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 ─ Class Weights for Imbalance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 2: Class Weights for Imbalance")
print("="*55)

from sklearn.utils.class_weight import compute_class_weight

n_classes = len(np.unique(y_train))
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(n_classes),
    y=y_train
)

print("Class weights (balanced):")
for cls_idx, weight in enumerate(class_weights):
    label_name = EMOTION_LABELS.get(cls_idx, str(cls_idx))
    count = np.sum(y_train == cls_idx)
    print(f"  {cls_idx} ({label_name:<10}): weight={weight:.4f}  count={count}")

np.save('data/processed/class_weights.npy', class_weights)
print("Saved: data/processed/class_weights.npy")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 ─ TF-IDF Vectorization
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 3: TF-IDF Vectorization")
print("="*55)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=20_000,        # vocabulary cap
    ngram_range=(1, 2),         # unigrams + bigrams
    min_df=3,                   # ignore very rare terms
    max_df=0.95,                # ignore very common terms
    sublinear_tf=True,          # log-scale TF dampening
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',  # only alphabetic tokens
)

print("Fitting TF-IDF on training data...")
X_train_tfidf = tfidf.fit_transform(train['text'])
X_val_tfidf   = tfidf.transform(val['text'])
X_test_tfidf  = tfidf.transform(test['text'])

print(f"Vocabulary size : {len(tfidf.vocabulary_):,}")
print(f"X_train shape   : {X_train_tfidf.shape}")
print(f"X_val shape     : {X_val_tfidf.shape}")
print(f"X_test shape    : {X_test_tfidf.shape}")
print(f"Density (train) : {X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]):.5f}")

save_npz('data/processed/X_train_tfidf.npz', X_train_tfidf)
save_npz('data/processed/X_val_tfidf.npz',   X_val_tfidf)
save_npz('data/processed/X_test_tfidf.npz',  X_test_tfidf)

with open('data/processed/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Saved: X_train_tfidf.npz, X_val_tfidf.npz, X_test_tfidf.npz")
print("Saved: data/processed/tfidf_vectorizer.pkl")

# Top TF-IDF feature names per class
print("\nTop 10 TF-IDF features per emotion:")
feature_names = np.array(tfidf.get_feature_names_out())
for cls_idx in range(n_classes):
    label_name = EMOTION_LABELS.get(cls_idx, str(cls_idx))
    class_mask = (y_train == cls_idx)
    mean_tfidf = np.asarray(X_train_tfidf[class_mask].mean(axis=0)).flatten()
    top_idx = mean_tfidf.argsort()[-10:][::-1]
    top_words = feature_names[top_idx]
    print(f"  {label_name:<10}: {', '.join(top_words)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 ─ Tokenization for Deep Learning
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 4: Tokenization for Deep Learning")
print("="*55)

MAX_LEN = 128  # standard for sentence-level tasks

try:
    from transformers import AutoTokenizer

    MODEL_NAME = 'distilbert-base-uncased'
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def batch_tokenize(texts, tokenizer, max_len, split_name):
        """Tokenize a list of texts and return a dict of numpy arrays."""
        print(f"  Tokenizing {split_name} ({len(texts):,} samples)...")
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='np'
        )
        return {
            'input_ids':      np.array(encoded['input_ids'],      dtype=np.int32),
            'attention_mask': np.array(encoded['attention_mask'], dtype=np.int8),
        }

    tok_train = batch_tokenize(train['text'], tokenizer, MAX_LEN, 'Train')
    tok_val   = batch_tokenize(val['text'],   tokenizer, MAX_LEN, 'Val')
    tok_test  = batch_tokenize(test['text'],  tokenizer, MAX_LEN, 'Test')

    print(f"  input_ids shape (train): {tok_train['input_ids'].shape}")

    for split_name, tok_data in [('train', tok_train), ('val', tok_val), ('test', tok_test)]:
        out_path = f'data/processed/tokenized_{split_name}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(tok_data, f)
        print(f"  Saved: {out_path}")

    print(f"\n  Tokenizer: {MODEL_NAME}, max_length={MAX_LEN}")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

except ImportError:
    # Fallback: simple whitespace tokenizer with vocabulary
    print("  transformers not installed — using simple word-index tokenizer.")
    print("  (Install transformers for BERT-style tokenization: pip install transformers)")

    from sklearn.feature_extraction.text import CountVectorizer

    # Build vocabulary from training set
    count_vec = CountVectorizer(max_features=10_000, token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
    count_vec.fit(train['text'])
    vocab = count_vec.vocabulary_   # word -> index (1-based after +1)

    def simple_tokenize(texts, vocab, max_len):
        """Convert texts to padded sequences of word indices."""
        sequences = []
        for text in texts:
            tokens = text.lower().split()
            ids = [vocab.get(t, 0) + 1 for t in tokens]  # 0 = unknown
            if len(ids) >= max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [0] * (max_len - len(ids))    # pad
            sequences.append(ids)
        return {'input_ids': np.array(sequences, dtype=np.int32)}

    tok_train = simple_tokenize(train['text'], vocab, MAX_LEN)
    tok_val   = simple_tokenize(val['text'],   vocab, MAX_LEN)
    tok_test  = simple_tokenize(test['text'],  vocab, MAX_LEN)

    print(f"  input_ids shape (train): {tok_train['input_ids'].shape}")

    for split_name, tok_data in [('train', tok_train), ('val', tok_val), ('test', tok_test)]:
        out_path = f'data/processed/tokenized_{split_name}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(tok_data, f)
        print(f"  Saved: {out_path}")

    # Save vocabulary (convert numpy int64 values to plain int for JSON)
    with open('data/processed/word_vocab.json', 'w') as f:
        json.dump({k: int(v) for k, v in vocab.items()}, f)
    print(f"  Saved: data/processed/word_vocab.json (vocab size={len(vocab):,})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 ─ Feature Engineering Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 5: Writing feature summary")
print("="*55)

summary_lines = [
    "Feature Engineering Summary",
    "=" * 55,
    "",
    "Dataset sizes (after cleaning):",
    f"  Train : {len(train):>7,}",
    f"  Val   : {len(val):>7,}",
    f"  Test  : {len(test):>7,}",
    "",
    "TF-IDF Vectorization:",
    f"  max_features  = 20,000",
    f"  ngram_range   = (1, 2)",
    f"  min_df        = 3",
    f"  max_df        = 0.95",
    f"  sublinear_tf  = True",
    f"  Actual vocab  = {len(tfidf.vocabulary_):,}",
    f"  X_train shape = {X_train_tfidf.shape}",
    "",
    "Tokenization for Deep Learning:",
    f"  max_length    = {MAX_LEN}",
    f"  Padding       = max_length (right-pad with 0)",
    f"  Truncation    = True",
    "",
    "Label Encoding:",
    f"  Classes       = {EMOTION_LABELS}",
    "",
    "Class Weights (balanced):",
]
for cls_idx, weight in enumerate(class_weights):
    label_name = EMOTION_LABELS.get(cls_idx, str(cls_idx))
    count = int(np.sum(y_train == cls_idx))
    summary_lines.append(f"  {cls_idx} {label_name:<10}: {weight:.4f}  (n={count})")

summary_lines += [
    "",
    "Output files:",
    "  data/processed/X_train_tfidf.npz",
    "  data/processed/X_val_tfidf.npz",
    "  data/processed/X_test_tfidf.npz",
    "  data/processed/tfidf_vectorizer.pkl",
    "  data/processed/tokenized_train.pkl",
    "  data/processed/tokenized_val.pkl",
    "  data/processed/tokenized_test.pkl",
    "  data/processed/label_encoder.pkl",
    "  data/processed/class_weights.npy",
]

summary_text = '\n'.join(summary_lines)
with open('data/processed/feature_summary.txt', 'w') as f:
    f.write(summary_text)

print(summary_text)
print("\n✓ Feature engineering complete.")
