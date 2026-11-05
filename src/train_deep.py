"""
Deep Learning Model Training Script (BiLSTM)
Template for Member 2 to complete hyperparameter tuning and training.

TODO for Member 2:
1. Complete the training loop with validation
2. Add early stopping logic
3. Implement hyperparameter tuning experiments
4. Save training curves (loss/accuracy plots)
5. Log all experiments to experiments/hyperparameter_log.csv
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.deep_model import BiLSTMEmotionClassifier
from src.utils.data_loader import load_emotion_data, calculate_class_weights
from src.utils.preprocessing import TextPreprocessor


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification"""
    
    def __init__(self, sequences, lengths, labels):
        self.sequences = torch.LongTensor(sequences)
        self.lengths = torch.LongTensor(lengths)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


def main():
    print("="*60)
    print("DEEP LEARNING MODEL TRAINING: BiLSTM")
    print("="*60)
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # =====================================================
    # HYPERPARAMETERS - TODO: Member 2 should tune these
    # =====================================================
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    MAX_LENGTH = 100
    
    # =====================================================
    # 1. Load and preprocess data
    # =====================================================
    print("\n[1/6] Loading data...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_emotion_data()
    
    # Calculate class weights
    weight_dict, class_weights = calculate_class_weights(train_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # =====================================================
    # 2. Text preprocessing
    # =====================================================
    print("\n[2/6] Preprocessing text...")
    preprocessor = TextPreprocessor(max_vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)
    
    # Build vocabulary on training data
    preprocessor.build_vocab(train_texts)
    
    # Transform all splits
    train_padded, train_lengths = preprocessor.transform(train_texts)
    val_padded, val_lengths = preprocessor.transform(val_texts)
    test_padded, test_lengths = preprocessor.transform(test_texts)
    
    print(f"✓ Train sequences: {train_padded.shape}")
    print(f"✓ Val sequences: {val_padded.shape}")
    print(f"✓ Test sequences: {test_padded.shape}")
    
    # =====================================================
    # 3. Create data loaders
    # =====================================================
    print("\n[3/6] Creating data loaders...")
    train_dataset = EmotionDataset(train_padded, train_lengths, train_labels)
    val_dataset = EmotionDataset(val_padded, val_lengths, val_labels)
    test_dataset = EmotionDataset(test_padded, test_lengths, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    
    # =====================================================
    # 4. Initialize model
    # =====================================================
    print("\n[4/6] Initializing model...")
    model = BiLSTMEmotionClassifier(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=6,
        dropout=DROPOUT
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"✓ Model initialized")
    print(f"  Parameters: ~{sum(p.numel() for p in model.parameters()):,}")
    
    # =====================================================
    # 5. Training loop - TODO: Member 2 complete this
    # =====================================================
    print("\n[5/6] Training model...")
    print("-"*60)
    
    # TODO: Member 2 - Add these features:
    # - Training history tracking (loss, accuracy per epoch)
    # - Validation after each epoch
    # - Early stopping based on validation loss
    # - Best model checkpointing
    # - Progress visualization with tqdm
    
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, lengths, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # TODO: Member 2 - Add validation phase here
        # Hint: Set model.eval(), use torch.no_grad()
        
        # Print epoch results
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f}")
        
        # TODO: Member 2 - Add early stopping logic
        # TODO: Member 2 - Save best model checkpoint
    
    # =====================================================
    # 6. Evaluation - TODO: Member 2 complete this
    # =====================================================
    print("\n[6/6] Evaluating model...")
    print("-"*60)
    
    # TODO: Member 2 - Complete evaluation on test set
    # - Generate predictions
    # - Calculate metrics (accuracy, F1, etc.)
    # - Save predictions to CSV
    # - Save metrics to JSON
    # - Generate confusion matrix
    
    print("\n" + "="*60)
    print("TRAINING TEMPLATE COMPLETE")
    print("="*60)
    print("\nTODO for Member 2:")
    print("  1. Complete validation loop in training")
    print("  2. Add early stopping logic")
    print("  3. Implement hyperparameter tuning (test different values)")
    print("  4. Save training curves (plot loss/accuracy)")
    print("  5. Log all experiments to experiments/hyperparameter_log.csv")
    print("  6. Complete test evaluation and save results")
    print("  7. Save best model as models/best_deep_model.pth")
    print("\n")


if __name__ == '__main__':
    main()
