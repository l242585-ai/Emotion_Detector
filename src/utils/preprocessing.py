"""
Text Preprocessing Utilities
Handles tokenization, vocabulary building, and sequence preparation for deep learning models.
"""

import numpy as np
from collections import Counter
import re


class TextPreprocessor:
    """
    Preprocessor for converting text to sequences for deep learning models.
    
    Args:
        max_vocab_size (int): Maximum vocabulary size (default: 10000)
        max_length (int): Maximum sequence length (default: 100)
        oov_token (str): Out-of-vocabulary token (default: '<UNK>')
        pad_token (str): Padding token (default: '<PAD>')
    """
    
    def __init__(self, max_vocab_size=10000, max_length=100, oov_token='<UNK>', pad_token='<PAD>'):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.pad_token = pad_token
        
        self.word2idx = {pad_token: 0, oov_token: 1}
        self.idx2word = {0: pad_token, 1: oov_token}
        self.vocab_size = 2  # Start with PAD and UNK
        
        self.is_fitted = False
    
    def clean_text(self, text):
        """
        Basic text cleaning.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\'\-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Simple whitespace tokenization.
        
        Args:
            text (str): Input text
        
        Returns:
            list: List of tokens
        """
        text = self.clean_text(text)
        tokens = text.split()
        return tokens
    
    def build_vocab(self, texts):
        """
        Build vocabulary from training texts.
        
        Args:
            texts (list): List of training texts
        
        Returns:
            self
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Tokenize all texts and count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        print(f"  Total unique words: {len(word_counts)}")
        
        # Get most common words (excluding PAD and UNK which are already added)
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # Build vocabulary
        for word, count in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        self.is_fitted = True
        
        print(f"  ✓ Vocabulary size: {self.vocab_size}")
        print(f"  ✓ Most common words: {[word for word, _ in most_common[:10]]}")
        
        return self
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences of integers.
        
        Args:
            texts (list): List of texts
        
        Returns:
            list: List of sequences (lists of integers)
        """
        if not self.is_fitted:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        sequences = []
        oov_idx = self.word2idx[self.oov_token]
        
        for text in texts:
            tokens = self.tokenize(text)
            sequence = [self.word2idx.get(token, oov_idx) for token in tokens]
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(self, sequences, max_length=None):
        """
        Pad sequences to the same length.
        
        Args:
            sequences (list): List of sequences
            max_length (int, optional): Maximum length (uses self.max_length if None)
        
        Returns:
            np.ndarray: Padded sequences, shape (n_samples, max_length)
            np.ndarray: Original lengths before padding
        """
        if max_length is None:
            max_length = self.max_length
        
        padded = np.zeros((len(sequences), max_length), dtype=np.int32)
        lengths = np.zeros(len(sequences), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            lengths[i] = length
            padded[i, :length] = seq[:length]
        
        return padded, lengths
    
    def transform(self, texts):
        """
        Transform texts to padded sequences (complete pipeline).
        
        Args:
            texts (list): List of texts
        
        Returns:
            np.ndarray: Padded sequences
            np.ndarray: Original lengths
        """
        sequences = self.texts_to_sequences(texts)
        padded, lengths = self.pad_sequences(sequences)
        return padded, lengths
    
    def fit_transform(self, texts):
        """
        Build vocabulary and transform texts in one step.
        
        Args:
            texts (list): List of training texts
        
        Returns:
            np.ndarray: Padded sequences
            np.ndarray: Original lengths
        """
        self.build_vocab(texts)
        return self.transform(texts)


def build_vocab(texts, max_vocab_size=10000):
    """
    Build vocabulary from texts (standalone function).
    
    Args:
        texts (list): List of texts
        max_vocab_size (int): Maximum vocabulary size
    
    Returns:
        dict: word2idx mapping
        dict: idx2word mapping
    """
    preprocessor = TextPreprocessor(max_vocab_size=max_vocab_size)
    preprocessor.build_vocab(texts)
    return preprocessor.word2idx, preprocessor.idx2word


def tokenize_and_pad(texts, word2idx, max_length=100, oov_token='<UNK>'):
    """
    Tokenize and pad texts using existing vocabulary (standalone function).
    
    Args:
        texts (list): List of texts
        word2idx (dict): Word to index mapping
        max_length (int): Maximum sequence length
        oov_token (str): Out-of-vocabulary token
    
    Returns:
        np.ndarray: Padded sequences
        np.ndarray: Original lengths
    """
    oov_idx = word2idx.get(oov_token, 1)
    
    # Simple tokenization
    sequences = []
    for text in texts:
        text = text.lower()
        tokens = text.split()
        sequence = [word2idx.get(token, oov_idx) for token in tokens]
        sequences.append(sequence)
    
    # Padding
    padded = np.zeros((len(sequences), max_length), dtype=np.int32)
    lengths = np.zeros(len(sequences), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        lengths[i] = length
        padded[i, :length] = seq[:length]
    
    return padded, lengths


if __name__ == '__main__':
    # Test preprocessing
    sample_texts = [
        "I am so happy today!",
        "This is the worst day ever.",
        "I love spending time with my friends."
    ]
    
    print("Testing TextPreprocessor...")
    preprocessor = TextPreprocessor(max_vocab_size=100, max_length=20)
    
    # Build vocab and transform
    padded, lengths = preprocessor.fit_transform(sample_texts)
    
    print(f"\nPadded sequences shape: {padded.shape}")
    print(f"Lengths: {lengths}")
    print(f"\nFirst sequence: {padded[0]}")
    
    # Decode back
    decoded = [preprocessor.idx2word.get(idx, '<UNK>') for idx in padded[0] if idx != 0]
    print(f"Decoded: {' '.join(decoded)}")
