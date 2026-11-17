"""
Model Service - loads trained emotion models and runs inference.

Baseline  : models/best_baseline.pkl  (local)
BiLSTM    : models/best_bilstm.pth    (local)
DistilBERT: HuggingFace Hub           (downloaded on first run, cached after)

Member 1 (Annushay Fatima, 24L-2585) - Deliverable 3
"""

import os
import sys
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

EMOTION_NAMES  = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness" : "😢",
    "joy"     : "😊",
    "love"    : "❤️",
    "anger"   : "😠",
    "fear"    : "😨",
    "surprise": "😲",
}

# HuggingFace repo where DistilBERT weights are stored
# Format: "username/repo-name"
HF_DISTILBERT_REPO = "l242585-ai/emotion-distilbert"


class ModelService:

    def __init__(self, models_dir="models", data_dir="data/processed"):
        self.models_dir    = os.path.join(PROJECT_ROOT, models_dir)
        self.data_dir      = os.path.join(PROJECT_ROOT, data_dir)
        self.loaded_models = {}
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── loaders ──────────────────────────────────────────────────────────────

    def load_baseline(self):
        path = os.path.join(self.models_dir, "best_baseline.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baseline model not found at {path}")

        from src.models.baseline_model import BaselineModel
        model = BaselineModel.load(path)
        self.loaded_models["baseline"] = {"model": model, "type": "baseline"}
        print(f"Baseline loaded from {path}")

    def load_bilstm(self):
        # try best_bilstm.pth first, fall back to best_model.pth
        path = os.path.join(self.models_dir, "best_bilstm.pth")
        if not os.path.exists(path):
            path = os.path.join(self.models_dir, "best_model.pth")
        if not os.path.exists(path):
            raise FileNotFoundError("BiLSTM model not found in models/")

        from src.models.deep_model   import BiLSTMEmotionClassifier
        from src.utils.preprocessing import TextPreprocessor
        from src.utils.data_loader   import load_emotion_data

        cp  = torch.load(path, map_location=self.device, weights_only=False)
        cfg = cp["config"]

        model = BiLSTMEmotionClassifier(
            vocab_size    = cfg["vocab_size"],
            embedding_dim = cfg["embedding_dim"],
            hidden_dim    = cfg["hidden_dim"],
            num_layers    = cfg["num_layers"],
            num_classes   = cfg["num_classes"],
            dropout       = cfg["dropout"],
        ).to(self.device)
        model.load_state_dict(cp["model_state_dict"])
        model.eval()

        train_texts, _, _, _, _, _ = load_emotion_data(self.data_dir)
        preprocessor = TextPreprocessor(max_vocab_size=10000, max_length=100)
        preprocessor.build_vocab(train_texts)

        self.loaded_models["bilstm"] = {
            "model"       : model,
            "preprocessor": preprocessor,
            "type"        : "bilstm",
        }
        print(f"BiLSTM loaded from {path}")

    def load_distilbert(self):
        """
        Load DistilBERT from HuggingFace Hub.
        On first run this downloads the weights (~250 MB) and caches them.
        Subsequent runs use the cache (no internet needed).

        HF repo: HF_DISTILBERT_REPO constant above.
        To upload: see docs/DELIVERABLE_3_TASKS.md
        """
        from transformers import DistilBertTokenizer
        from src.models.distilbert_model import DistilBertEmotionClassifier

        print(f"Loading DistilBERT from HuggingFace: {HF_DISTILBERT_REPO}")

        # Try HuggingFace Hub first
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id  = HF_DISTILBERT_REPO,
                filename = "best_distilbert.pth",
            )
            cp  = torch.load(weights_path, map_location=self.device, weights_only=False)
            cfg = cp.get("config", {})

            model = DistilBertEmotionClassifier(
                num_classes = cfg.get("num_classes", 6),
                dropout     = cfg.get("dropout",     0.3),
            ).to(self.device)
            model.load_state_dict(cp["model_state_dict"])
            model.eval()

            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

            self.loaded_models["distilbert"] = {
                "model"     : model,
                "tokenizer" : tokenizer,
                "max_length": cfg.get("max_length", 128),
                "type"      : "distilbert",
            }
            print("DistilBERT loaded from HuggingFace Hub")
            return

        except Exception as e:
            print(f"HuggingFace load failed: {e}")

        # Fallback: local file
        local_path = os.path.join(self.models_dir, "best_distilbert.pth")
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                "DistilBERT not found locally or on HuggingFace Hub. "
                f"Upload weights to {HF_DISTILBERT_REPO} or place best_distilbert.pth in models/"
            )

        cp  = torch.load(local_path, map_location=self.device, weights_only=False)
        cfg = cp.get("config", {})

        model = DistilBertEmotionClassifier(
            num_classes = cfg.get("num_classes", 6),
            dropout     = cfg.get("dropout",     0.3),
        ).to(self.device)
        model.load_state_dict(cp["model_state_dict"])
        model.eval()

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        self.loaded_models["distilbert"] = {
            "model"     : model,
            "tokenizer" : tokenizer,
            "max_length": cfg.get("max_length", 128),
            "type"      : "distilbert",
        }
        print(f"DistilBERT loaded from local file {local_path}")

    def load_all(self):
        loaded = []
        for name, loader in [
            ("baseline",   self.load_baseline),
            ("bilstm",     self.load_bilstm),
            ("distilbert", self.load_distilbert),
        ]:
            try:
                loader()
                loaded.append(name)
            except Exception as e:
                print(f"Could not load {name}: {e}")
        return loaded

    def get_available_models(self):
        return list(self.loaded_models.keys())

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, text, model_name="distilbert"):
        if model_name not in self.loaded_models:
            raise ValueError(
                f"Model '{model_name}' not loaded. "
                f"Available: {self.get_available_models()}"
            )

        entry = self.loaded_models[model_name]
        t0    = time.perf_counter()

        if entry["type"] == "baseline":
            probs = entry["model"].predict_proba([text])[0]
        elif entry["type"] == "bilstm":
            probs = self._bilstm_probs(text, entry)
        else:
            probs = self._distilbert_probs(text, entry)

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

    def _bilstm_probs(self, text, entry):
        padded, lengths = entry["preprocessor"].transform([text])
        seq_t = torch.LongTensor(padded).to(self.device)
        len_t = torch.LongTensor(lengths).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(entry["model"](seq_t, len_t), dim=1).cpu().numpy()[0]
        return probs

    def _distilbert_probs(self, text, entry):
        enc = entry["tokenizer"](
            text,
            max_length     = entry["max_length"],
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            probs = torch.softmax(entry["model"](ids, mask), dim=1).cpu().numpy()[0]
        return probs
