"""
Data Loading Utilities
Loads cleaned emotion dataset and provides utilities for data handling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def load_emotion_data(data_dir='data/processed'):
    """
    Load cleaned emotion dataset from CSV files.
    
    This is the main API that Member 2 will use for training.
    
    Args:
        data_dir (str): Directory containing processed CSV files
    
    Returns:
        tuple: (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
               - texts are lists of strings
               - labels are numpy arrays of integers (0-5)
    """
    print("Loading emotion dataset...")
    
    # Load CSV files
    train_df = pd.read_csv(os.path.join(data_dir, 'train_cleaned.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_cleaned.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_cleaned.csv'))
    
    # Extract texts and labels
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].values
    
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].values
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].values
    
    print(f"  ✓ Training samples: {len(train_texts)}")
    print(f"  ✓ Validation samples: {len(val_texts)}")
    print(f"  ✓ Test samples: {len(test_texts)}")
    print(f"  ✓ Unique labels: {sorted(np.unique(train_labels))}")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def calculate_class_weights(labels, num_classes=6):
    """
    Calculate class weights for handling imbalanced dataset.
    
    Args:
        labels (array): Training labels
        num_classes (int): Number of classes (default: 6)
    
    Returns:
        dict: Dictionary mapping class index to weight
        np.ndarray: Array of class weights
    """
    # Calculate weights using sklearn
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    # Create dictionary
    weight_dict = {i: weights[i] for i in range(num_classes)}
    
    print("\nClass weights (for handling imbalance):")
    emotion_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    for i in range(num_classes):
        count = np.sum(labels == i)
        print(f"  Class {i} ({emotion_names[i]:8s}): weight={weights[i]:.3f}, count={count}")
    
    return weight_dict, weights


def get_label_distribution(labels, split_name='Train'):
    """
    Get distribution statistics for labels.
    
    Args:
        labels (array): Label array
        split_name (str): Name of the split (for display)
    
    Returns:
        dict: Dictionary with label counts and percentages
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    distribution = {}
    emotion_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    
    print(f"\n{split_name} Label Distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / total) * 100
        emotion = emotion_names[label] if label < len(emotion_names) else f"class_{label}"
        distribution[int(label)] = {'count': int(count), 'percentage': percentage}
        print(f"  {label} ({emotion:8s}): {count:5d} ({percentage:5.2f}%)")
    
    return distribution


def load_emotion_data_with_info(data_dir='data/processed', verbose=True):
    """
    Load emotion data with additional statistics and information.
    
    Args:
        data_dir (str): Directory containing processed CSV files
        verbose (bool): Print statistics if True
    
    Returns:
        dict: Dictionary containing data and metadata
    """
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_emotion_data(data_dir)
    
    if verbose:
        # Calculate class weights
        weight_dict, weights = calculate_class_weights(train_labels)
        
        # Get distributions
        train_dist = get_label_distribution(train_labels, 'Train')
        val_dist = get_label_distribution(val_labels, 'Validation')
        test_dist = get_label_distribution(test_labels, 'Test')
    else:
        weight_dict, weights = calculate_class_weights(train_labels)
        train_dist = val_dist = test_dist = None
    
    # Package everything
    data_info = {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels},
        'test': {'texts': test_texts, 'labels': test_labels},
        'class_weights': weights,
        'class_weight_dict': weight_dict,
        'num_classes': 6,
        'emotion_names': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
        'distributions': {
            'train': train_dist,
            'val': val_dist,
            'test': test_dist
        }
    }
    
    return data_info


if __name__ == '__main__':
    # Test the data loader
    print("Testing data loader...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_emotion_data()
    
    print(f"\nSample texts:")
    for i in range(3):
        print(f"  [{train_labels[i]}] {train_texts[i][:60]}...")
    
    # Calculate class weights
    weight_dict, weights = calculate_class_weights(train_labels)
