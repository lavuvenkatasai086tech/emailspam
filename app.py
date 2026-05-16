"""
Email Spam Detection — Flask Web Application
Run: python app.py
"""

import os
import re
import json
import string
import joblib

from flask import Flask, render_template, request, jsonify
from datetime import datetime

# ─── Built-in stopwords (no NLTK needed) ─────────────────────────────────────
STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their","theirs",
    "what","which","who","whom","this","that","am","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","a","an","the","and","but","if","or",
    "as","at","by","for","in","of","on","to","up","with","about","into","through",
    "during","before","after","above","below","between","each","than","so","then",
    "now","just","both","all","any","few","more","most","other","some","such","no",
    "nor","not","only","same","very","too","also","here","there","when","where",
    "how","again","further","once","from","out","off","over","under","while",
}

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("model", "spam_classifier.pkl")
METRICS_PATH = os.path.join("model", "metrics.json")

pipeline = None
metrics  = {}

def load_model():
    global pipeline, metrics
    if not os.path.exists(MODEL_PATH):
        print("Warning: Model not found. Run `python train_model.py` first.")
        return False
    pipeline = joblib.load(MODEL_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    print("Model loaded successfully.")
    return True


# ─── Preprocessing (matches train_model.py exactly) ──────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", " phonetoken ", text)
    text = re.sub(r"\b\d+\b", " numtoken ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def extract_features(text: str) -> dict:
    """Return heuristic spam signal features for UI display."""
    spam_keywords = [
        "free", "win", "winner", "won", "prize", "cash", "money", "earn",
        "guaranteed", "click", "urgent", "limited", "offer", "claim",
        "congratulations", "selected", "exclusive", "discount", "deal",
        "buy now", "act now", "apply now", "order now", "subscribe",
        "credit", "loan", "investment", "bitcoin", "crypto", "profit",
        "password", "account", "verify", "confirm", "suspended", "alert",
        "security", "hack", "nigeria", "prince", "inheritance", "million",
        "billion", "sex", "adult", "viagra", "casino", "poker", "bet",
    ]
    text_lower = text.lower()
    words      = text_lower.split()
    word_count = len(words)

    found_keywords = [kw for kw in spam_keywords if kw in text_lower]

    # Caps ratio
    caps_chars = sum(1 for c in text if c.isupper())
    total_chars = max(len(text), 1)
    caps_ratio  = round(caps_chars / total_chars * 100, 1)

    # Exclamations / question marks
    excl_count = text.count("!")
    url_count  = len(re.findall(r"https?://\S+|www\.\S+", text_lower))

    # Digit density
    digit_chars  = sum(1 for c in text if c.isdigit())
    digit_density = round(digit_chars / total_chars * 100, 1)

    return {
        "word_count":    word_count,
        "char_count":    len(text),
        "spam_keywords": found_keywords[:8],   # top 8 for display
        "caps_ratio":    caps_ratio,
        "exclamations":  excl_count,
        "url_count":     url_count,
        "digit_density": digit_density,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(silent=True) or {}
    email_text = data.get("text", "").strip()

    if not email_text:
        return jsonify({"error": "No email text provided."}), 400

    # Preprocess and predict
    clean = preprocess_text(email_text)
    if not clean:
        return jsonify({"error": "Email text is too short or contains only stop words."}), 400

    prediction   = pipeline.predict([clean])[0]
    probabilities = pipeline.predict_proba([clean])[0]

    # Class order: pipeline.classes_ = ['ham', 'spam']
    classes   = list(pipeline.classes_)
    spam_idx  = classes.index("spam")
    ham_idx   = classes.index("ham")

    spam_prob = float(probabilities[spam_idx])
    ham_prob  = float(probabilities[ham_idx])

    features = extract_features(email_text)

    # Risk level
    if spam_prob >= 0.85:
        risk = "high"
    elif spam_prob >= 0.55:
        risk = "medium"
    elif spam_prob >= 0.35:
        risk = "low"
    else:
        risk = "safe"

    return jsonify({
        "prediction":  prediction,       # "spam" or "ham"
        "is_spam":     prediction == "spam",
        "spam_prob":   round(spam_prob * 100, 2),
        "ham_prob":    round(ham_prob  * 100, 2),
        "confidence":  round(max(spam_prob, ham_prob) * 100, 2),
        "risk_level":  risk,
        "features":    features,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None,
        "metrics": metrics,
    })


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not load_model():
        print("Run `python train_model.py` first, then restart the app.")
    else:
        print("Starting Flask server at http://127.0.0.1:5000")
        app.run(debug=True, port=5000)
