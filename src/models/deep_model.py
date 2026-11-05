"""
BiLSTM Model for Emotion Classification

Architecture:
    Input (batch_size, seq_length) 
    → Embedding (vocab_size, embedding_dim=128)
    → BiLSTM (2 layers, hidden_dim=64, bidirectional)
    → Dropout (p=0.3)
    → Fully Connected (128 → 6)
    → Softmax (implicit in CrossEntropyLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMEmotionClassifier(nn.Module):
    """
    Bidirectional LSTM for emotion classification.
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings (default: 128)
        hidden_dim (int): Hidden dimension of LSTM (default: 64)
        num_layers (int): Number of LSTM layers (default: 2)
        num_classes (int): Number of emotion classes (default: 6)
        dropout (float): Dropout probability (default: 0.3)
        pretrained_embeddings (torch.Tensor, optional): Pre-trained embeddings
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2,
        num_classes=6,
        dropout=0.3,
        pretrained_embeddings=None
    ):
        super(BiLSTMEmotionClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Bidirectional LSTM
        # Note: bidirectional=True means output will be 2*hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        # Input: 2*hidden_dim (because bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of token indices, shape (batch_size, seq_length)
            lengths (torch.Tensor, optional): Actual lengths of sequences (for packing)
        
        Returns:
            torch.Tensor: Logits for each class, shape (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_length) → (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Pack sequences if lengths are provided (for efficiency)
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # BiLSTM: (batch_size, seq_length, embedding_dim) → (batch_size, seq_length, 2*hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Unpack if we packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Use the last hidden states from both directions
        # hidden shape: (num_layers*2, batch_size, hidden_dim)
        # We want the last layer's forward and backward hidden states
        forward_hidden = hidden[-2, :, :]  # Forward direction last layer
        backward_hidden = hidden[-1, :, :]  # Backward direction last layer
        
        # Concatenate forward and backward: (batch_size, 2*hidden_dim)
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Dropout
        dropped = self.dropout(final_hidden)
        
        # Fully connected: (batch_size, 2*hidden_dim) → (batch_size, num_classes)
        logits = self.fc(dropped)
        
        return logits
    
    def predict(self, x, lengths=None):
        """
        Make predictions (returns class indices).
        
        Args:
            x (torch.Tensor): Input tensor
            lengths (torch.Tensor, optional): Sequence lengths
        
        Returns:
            torch.Tensor: Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x, lengths=None):
        """
        Make probability predictions.
        
        Args:
            x (torch.Tensor): Input tensor
            lengths (torch.Tensor, optional): Sequence lengths
        
        Returns:
            torch.Tensor: Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


def create_model(vocab_size, config=None):
    """
    Factory function to create BiLSTM model with default or custom config.
    
    Args:
        vocab_size (int): Vocabulary size
        config (dict, optional): Configuration dictionary with hyperparameters
    
    Returns:
        BiLSTMEmotionClassifier: Initialized model
    """
    if config is None:
        config = {
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_classes': 6,
            'dropout': 0.3
        }
    
    model = BiLSTMEmotionClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 2),
        num_classes=config.get('num_classes', 6),
        dropout=config.get('dropout', 0.3)
    )
    
    return model
