"""Quick test for preprocessing utilities"""
from src.utils.preprocessing import TextPreprocessor
from src.utils.data_loader import load_emotion_data

print("Testing Preprocessing...")
train_texts, _, _, _, _, _ = load_emotion_data()

preprocessor = TextPreprocessor(max_vocab_size=10000, max_length=100)
preprocessor.build_vocab(train_texts)

# Test transform
padded, lengths = preprocessor.transform(train_texts[:5])
print(f"✓ Vocab size: {preprocessor.vocab_size}")
print(f"✓ Padded shape: {padded.shape}")
print(f"✓ Sample lengths: {lengths}")
print("✓ Preprocessing test passed!")
