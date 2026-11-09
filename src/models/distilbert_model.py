"""
DistilBERT Model for Emotion Classification

Architecture:
    Input (tokenized text with attention mask)
    → DistilBERT (pretrained, 6 layers, 768 hidden dim)
    → [CLS] token representation (768,)
    → Dropout (p=0.3)
    → Fully Connected (768 → 6)
    → Output: Logits for 6 emotions

Pretrained model: distilbert-base-uncased (66M params, 40% smaller than BERT)
Fine-tuned for 6-class emotion classification.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class DistilBertEmotionClassifier(nn.Module):
    """
    DistilBERT-based emotion classifier with a classification head.

    Args:
        num_classes (int): Number of emotion classes (default: 6)
        dropout (float): Dropout probability on [CLS] representation (default: 0.3)
        pretrained_model (str): HuggingFace model name (default: 'distilbert-base-uncased')
        freeze_base (bool): If True, freeze DistilBERT weights — only train classifier head.
                            Useful for fast experimentation. (default: False)
    """

    def __init__(
        self,
        num_classes=6,
        dropout=0.3,
        pretrained_model='distilbert-base-uncased',
        freeze_base=False
    ):
        super(DistilBertEmotionClassifier, self).__init__()

        self.num_classes = num_classes
        self.pretrained_model = pretrained_model

        # Load pretrained DistilBERT backbone
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)

        # Optionally freeze the backbone (useful for hyperparameter search on head only)
        if freeze_base:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        # Classification head on top of [CLS] token
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Token IDs, shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Attention mask, shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Logits for each class, shape (batch_size, num_classes)
        """
        # DistilBERT forward pass
        # outputs.last_hidden_state: (batch_size, seq_length, 768)
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract [CLS] token representation (first token)
        # Shape: (batch_size, 768)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Classification head
        x = self.pre_classifier(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits

    def predict(self, input_ids, attention_mask):
        """
        Make predictions (returns class indices).

        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask

        Returns:
            torch.Tensor: Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, input_ids, attention_mask):
        """
        Make probability predictions using softmax.

        Returns:
            torch.Tensor: Class probabilities, shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        return probabilities


class EmotionDatasetBERT(torch.utils.data.Dataset):
    """
    PyTorch Dataset for DistilBERT emotion classification.

    Args:
        texts (list): List of raw text strings
        labels (array): Array of integer labels (0-5)
        tokenizer: HuggingFace tokenizer
        max_length (int): Max token length (default: 128)
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize with DistilBERT tokenizer
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),       # (max_length,)
            'attention_mask': encoding['attention_mask'].squeeze(0),  # (max_length,)
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_distilbert_model(num_classes=6, dropout=0.3, freeze_base=False):
    """
    Factory function to create DistilBERT classifier and tokenizer.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
        freeze_base (bool): Whether to freeze the DistilBERT backbone

    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertEmotionClassifier(
        num_classes=num_classes,
        dropout=dropout,
        freeze_base=freeze_base
    )
    return model, tokenizer
