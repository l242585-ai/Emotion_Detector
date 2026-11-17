"""
Model Service — Loads trained emotion models and runs inference.

Supports:
  1. Baseline  : TF-IDF + Logistic Regression  (models/best_baseline.pkl)
  2. BiLSTM    : Bidirectional LSTM             (models/best_bilstm.pth)
  3. DistilBERT: Fine-tuned transformer         (models/best_model.pth)

Member 1 (Annushay Fatima, 24L-2585) — Deliverable 3
"""

import os
import sys
import time
import pickle
import logging
import numpy as np
import torch

# ── Add project root so src.* imports work ──────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ── Emotion metadata ─────────────────────────────────────────────────────────
EMOTION_NAMES  = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness" : "😢",
    "joy"     : "😊",
    "love"    : "❤️",
    "anger"   : "😠",
    "fear"    : "😨",
    "surprise": "😲",
}


class ModelService:
    """
    Unified service that loads any of the three trained models
    and provides a consistent predict() interface.

    Usage:
        svc = ModelService()
        svc.load_all()          # or load individual models
        result = svc.predict("I feel fantastic!", model_name="distilbert")
    """

    def __init__(self, models_dir: str = "models", data_dir: str = "data/processed"):
        self.models_dir     = os.path.join(PROJECT_ROOT, models_dir)
        self.data_dir       = os.path.join(PROJECT_ROOT, data_dir)
        self.loaded_models  : dict = {}
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ModelService initialised. Device: %s", self.device)

    # ── Model loading ────────────────────────────────────────────────────────

    def load_baseline(self) -> None:
        """Load the TF-IDF + Logistic Regression baseline model."""
        path = os.path.join(self.models_dir, "best_baseline.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Baseline model not found at {path}. "
                "Run: python src/train_baseline.py"
            )

        from src.models.baseline_model import BaselineModel
        model = BaselineModel.load(path)
        self.loaded_models["baseline"] = {"model": model, "type": "baseline"}
        logger.info("Baseline model loaded from %s", path)

    def load_bilstm(self) -> None:
        """Load the BiLSTM model and rebuild its preprocessor from training data."""
        path = os.path.join(self.models_dir, "best_bilstm.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BiLSTM model not found at {path}. "
                "Run: python src/train_deep.py"
            )

        from src.models.deep_model    import BiLSTMEmotionClassifier
        from src.utils.preprocessing  import TextPreprocessor
        from src.utils.data_loader    import load_emotion_data

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg        = checkpoint["config"]

        model = BiLSTMEmotionClassifier(
            vocab_size     = cfg["vocab_size"],
            embedding_dim  = cfg["embedding_dim"],
            hidden_dim     = cfg["hidden_dim"],
            num_layers     = cfg["num_layers"],
            num_classes    = cfg["num_classes"],
            dropout        = cfg["dropout"],
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Rebuild the same vocabulary the model was trained on
        train_texts, _, _, _, _, _ = load_emotion_data(self.data_dir)
        preprocessor = TextPreprocessor(max_vocab_size=10000, max_length=100)
        preprocessor.build_vocab(train_texts)

        self.loaded_models["bilstm"] = {
            "model"       : model,
            "preprocessor": preprocessor,
            "type"        : "bilstm",
        }
        logger.info("BiLSTM model loaded from %s", path)

    def load_distilbert(self) -> None:
        """Load the fine-tuned DistilBERT model."""
        path = os.path.join(self.models_dir, "best_model.pth")
        if not os.path.exists(path):
            # Try alternative filename
            path = os.path.join(self.models_dir, "best_distilbert.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "DistilBERT model not found. "
                "Run: python src/train_distilbert.py"
            )

        from src.models.distilbert_model import DistilBertEmotionClassifier
        from transformers                import DistilBertTokenizer

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg        = checkpoint.get("config", {})

        model = DistilBertEmotionClassifier(
            num_classes = cfg.get("num_classes", 6),
            dropout     = cfg.get("dropout",     0.3),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        self.loaded_models["distilbert"] = {
            "model"     : model,
            "tokenizer" : tokenizer,
            "max_length": cfg.get("max_length", 128),
            "type"      : "distilbert",
        }
        logger.info("DistilBERT model loaded from %s", path)

    def load_all(self) -> list:
        """
        Attempt to load all three models.
        Models that are missing or fail to load are skipped with a warning.

        Returns:
            list: Names of successfully loaded models.
        """
        loaded = []
        for name, loader in [
            ("baseline",   self.load_baseline),
            ("bilstm",     self.load_bilstm),
            ("distilbert", self.load_distilbert),
        ]:
            try:
                loader()
                loaded.append(name)
            except Exception as exc:
                logger.warning("Could not load %s: %s", name, exc)
        return loaded

    def get_available_models(self) -> list:
        """Return the names of currently loaded models."""
        return list(self.loaded_models.keys())

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, text: str, model_name: str = "distilbert") -> dict:
        """
        Run inference on a single text string.

        Args:
            text (str)       : Input text (1 – 5 000 characters).
            model_name (str) : One of 'baseline', 'bilstm', 'distilbert'.

        Returns:
            dict: {
                'emotion'      : str,
                'emotion_id'   : int,
                'emoji'        : str,
                'confidence'   : float,
                'probabilities': {emotion: prob},
                'model'        : str,
                'latency_ms'   : float,
            }

        Raises:
            ValueError : If the requested model is not loaded.
        """
        if model_name not in self.loaded_models:
            raise ValueError(
                f"Model '{model_name}' is not loaded. "
                f"Available: {self.get_available_models()}"
            )

        entry = self.loaded_models[model_name]
        t0    = time.perf_counter()

        dispatch = {
            "baseline"  : self._predict_baseline,
            "bilstm"    : self._predict_bilstm,
            "distilbert": self._predict_distilbert,
        }
        probs = dispatch[entry["type"]](text, entry)

        latency_ms = (time.perf_counter() - t0) * 1000
        pred_id    = int(np.argmax(probs))
        emotion    = EMOTION_NAMES[pred_id]

        return {
            "emotion"      : emotion,
            "emotion_id"   : pred_id,
            "emoji"        : EMOTION_EMOJIS[emotion],
            "confidence"   : float(probs[pred_id]),
            "probabilities": {EMOTION_NAMES[i]: float(probs[i]) for i in range(6)},
            "model"        : model_name,
            "latency_ms"   : round(latency_ms, 2),
        }

    # ── Private inference helpers ────────────────────────────────────────────

    def _predict_baseline(self, text: str, entry: dict) -> np.ndarray:
        return entry["model"].predict_proba([text])[0]

    def _predict_bilstm(self, text: str, entry: dict) -> np.ndarray:
        padded, lengths = entry["preprocessor"].transform([text])
        seq_t = torch.LongTensor(padded).to(self.device)
        len_t = torch.LongTensor(lengths).to(self.device)

        with torch.no_grad():
            logits = entry["model"](seq_t, len_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def _predict_distilbert(self, text: str, entry: dict) -> np.ndarray:
        enc = entry["tokenizer"](
            text,
            max_length  = entry["max_length"],
            padding     = "max_length",
            truncation  = True,
            return_tensors = "pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = entry["model"](input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs
