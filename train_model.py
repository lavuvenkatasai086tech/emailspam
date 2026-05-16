"""
Email Spam Detection - Model Training Script
TF-IDF vectorization + Logistic Regression.
No NLTK network downloads needed — uses built-in regex preprocessing.
Run once: python train_model.py
"""

import os
import re
import json
import string
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

# ─── Simple stopwords (no NLTK needed) ───────────────────────────────────────
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

# ─── Text Preprocessing (pure Python, no NLTK) ───────────────────────────────
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

# ─── Dataset ──────────────────────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    spam = [
        "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize now! Limited time offer!",
        "FREE money! Get rich quick! Make $5000 per week working from home! No experience needed!",
        "You have been selected for an exclusive offer! Claim your free iPhone 15 now! Act fast!",
        "URGENT: Your bank account has been compromised. Click immediately to verify your details!",
        "Win a luxury vacation worth $10,000! You're our lucky winner! Respond now!",
        "Investment opportunity! Double your money in 30 days guaranteed! Risk free!",
        "Get paid $500 daily from home! Legitimate work! No skills required! Apply now!",
        "Lottery winner notification! You have won 2 million dollars! Send us your details!",
        "FINAL NOTICE: Your package is on hold. Pay $2.99 clearance fee to release it!",
        "Exclusive VIP offer just for you! 90% discount on luxury watches! Buy now!",
        "Make money online fast! Our proven system generates $10,000 monthly! Join free!",
        "You owe back taxes! IRS urgent notice! Call this number immediately or face arrest!",
        "FREE gift card worth $500! Complete this survey and claim yours today!",
        "Bitcoin investment! 300% returns guaranteed! Invest now before it's too late!",
        "Nigerian prince needs your help! Share in $50 million fortune! Reply urgently!",
        "Lose 30 pounds in 30 days! Miracle weight loss pill! Doctor approved! Order now!",
        "Your PayPal account is suspended! Verify immediately to avoid permanent closure!",
        "Hot singles in your area want to meet you! Click here to see their profiles!",
        "Cheap Viagra! Buy online without prescription! Discreet shipping guaranteed!",
        "Work from home! Earn $1000 daily! Amazon approved! No experience needed apply now!",
        "WINNER ALERT! You've been chosen! Claim $250 Amazon gift card before it expires!",
        "Debt consolidation! Eliminate all debt immediately! Bad credit OK! Call now!",
        "Free credit score check! You qualify for $50,000 personal loan! Apply instantly!",
        "Urgent security alert! Your Microsoft account has been hacked! Fix it immediately!",
        "Earn passive income! Our automated system works 24/7! Make thousands weekly!",
        "SPECIAL PROMOTION: Buy 1 get 3 free! Limited stock! Don't miss this incredible deal!",
        "You have a secret admirer! Find out who likes you! Click here now!",
        "Shocking diet trick! Doctors hate it! Lose belly fat overnight! Click to learn more!",
        "Your inheritance fund of $2.5 million is ready for transfer! Contact us immediately!",
        "CASINO BONUS: Get $1000 free chips! No deposit required! Play and win big tonight!",
        "Exclusive member discount! 80% off all products today only! Shop now before midnight!",
        "Verify your identity now or your account will be deleted within 24 hours!",
        "Pre-approved for $25,000 loan! No credit check! Funds deposited same day!",
        "Business opportunity! Become your own boss! Unlimited earning potential guaranteed!",
        "Limited time: Buy 2 supplements get 5 free! Anti-aging miracle formula! Order today!",
        "Congratulations your email was selected! Redeem your $750 reward card instantly!",
        "Alert! Suspicious login detected on your account! Click here to secure it now!",
        "Stock tip! This penny stock will explode! Buy immediately before announcement!",
        "Prescription drugs at 90% discount! No doctor needed! Worldwide shipping available!",
        "You qualify for government grant money! $9,000 free! Never needs to be repaid!",
        "FLASH SALE: Rolex watches at 95% off! Authentic luxury! Order while stocks last!",
        "Text STOP to unsubscribe. Claim your free holiday prize! Winner confirmed today!",
        "Your car warranty is expiring! Renew now to avoid massive repair bills!",
        "Earn $5000 weekly filling out surveys! Companies pay big for your opinion!",
        "MLM opportunity! Ground floor! Join our team and achieve financial freedom now!",
        "FREE trial! Premium dating site membership! Meet beautiful people near you today!",
        "Unclaimed prize money in your name! $15,000 waiting! Claim before it's forfeited!",
        "Click to win! You are today's selected visitor! Claim your smartphone prize now!",
        "Hair loss cure discovered! Regrow full head of hair in 30 days! 100% natural!",
        "ACCOUNT SUSPENDED: Update your payment method immediately to restore access!",
        "Mega sale happening now! Everything must go! 70% discount sitewide! Shop immediately!",
        "u r winner! ur mobile number has won prize! claim now txt WIN to 87121",
        "FREE msg: Get ur ringtone 4 free! Reply YES to 84484 now!",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt CSH11 and send to 87575",
        "URGENT! Your Mobile No was awarded a Bonus Caller Prize call to claim!",
        "You have won a guaranteed prize of cash. Text YES to claim your reward now!",
        "Had your mobile 11 months or more? You are entitled to update to latest colour mobiles for Free!",
        "FREE entry in weekly competition to win FA Cup final tickets! Text to enter now!",
        "PRIVATE! Your Account Statement shows 800 un-redeemed points. Call to redeem!",
        "Congratulations! You have been specially selected to receive our exclusive bonus offer!",
        "Your email account wins $500 weekly draw! Reply with your details to claim prize money!",
        "Special offer: 100% free access to premium adult content! Click here to view now!",
        "Make $200-$500 daily from home! Proven system! Thousands already earning! Join now!",
    ]

    ham = [
        "Hi team, please find attached the quarterly report. Let me know if you have any questions.",
        "The meeting is scheduled for 3 PM tomorrow. Please confirm your attendance by end of day.",
        "Could you review the project proposal and send me your feedback by Friday?",
        "I wanted to follow up on our last conversation about the budget allocation for Q3.",
        "Thanks for sending over the contract. I'll have our legal team review it this week.",
        "Can we reschedule our 1:1 to Thursday? I have a conflict on Wednesday afternoon.",
        "The client presentation went well! They loved the new design direction we proposed.",
        "Please update your timesheet by end of business today. HR needs all submissions complete.",
        "Here are the meeting notes from today's standup. Action items are highlighted in yellow.",
        "I'm OOO next week. For urgent matters, please reach out to Sarah or James instead.",
        "The server maintenance window is scheduled for Sunday 2-4 AM. Please save your work.",
        "Welcome to the team! Your onboarding documents are attached. Start date is June 1st.",
        "We need to discuss the roadmap priorities for next quarter. Are you free Thursday?",
        "The bug has been fixed and deployed to production. Monitoring shows all systems normal.",
        "Your annual performance review is next Monday. Please complete the self-assessment form.",
        "Great work on the presentation today! The stakeholders were very impressed with your analysis.",
        "I've reviewed your pull request and left some comments. Overall it looks good to merge.",
        "Can you send me the client contact details? I need to schedule a call for next week.",
        "The invoice has been processed and payment will be released within 5 business days.",
        "Thanks for covering the support tickets while I was out. Really appreciate the help!",
        "Hey! Are you free this weekend? We're having a barbecue at our place on Saturday.",
        "Happy birthday! Hope you have an amazing day filled with joy and celebration!",
        "Thanks for dinner last night! The food was absolutely delicious. We should do it again soon.",
        "Just checking in to see how you're feeling after the surgery. Let me know if you need anything.",
        "Did you see the game last night? That last-minute goal was absolutely incredible!",
        "Mom wants to know if you're coming home for the holidays. Let me know your plans!",
        "The photos from the wedding came out beautifully! I'll share the album link with you.",
        "Can you recommend a good plumber? Our kitchen sink has been leaking for a week.",
        "Just finished that book you recommended. You were right, it was absolutely brilliant!",
        "We're thinking of adopting a dog! Any advice for first-time pet owners?",
        "The kids had a great time at the birthday party. Thank you so much for the invitation!",
        "I found a great new coffee shop near the office. We should grab breakfast there sometime.",
        "How was your vacation? I've been looking forward to hearing all about your trip to Italy.",
        "Just a reminder about our hiking trip this Sunday. Meet at the trailhead at 7 AM.",
        "Congratulations on your promotion! All your hard work is really paying off!",
        "We're moving to a new apartment next month. Any tips for making the move smoother?",
        "I'm trying a new recipe tonight. Moroccan lamb with roasted vegetables. Wish me luck!",
        "Your weekly newsletter from Tech Digest. Top stories in AI, cloud computing, and startups.",
        "Your order #12345 has been shipped! Estimated delivery is Thursday, May 14th.",
        "Your subscription renewal is coming up. Here are your options for the next billing cycle.",
        "Your doctor's appointment is confirmed for May 20th at 2:30 PM. Reply to reschedule.",
        "Your flight to New York departs at 6:45 AM. Check in is now open. Download boarding pass.",
        "Your account statement for April is now available. Log in to view your transactions.",
        "Reminder: Your library book Deep Learning is due back on May 18th.",
        "Your package has been delivered to your front door. Photo confirmation attached.",
        "Your gym membership renews on June 1st. Update payment details if needed.",
        "Class registration for Fall 2026 opens Monday. Check your student portal for details.",
        "Your tax documents are ready to download in your account dashboard.",
        "Booking confirmation: Hotel reservation for June 15-18. Check-in at 3 PM.",
        "Your password was changed successfully. If this wasn't you, contact support immediately.",
        "Monthly report: Your energy usage decreased by 12% compared to last month. Great job!",
        "GitHub: New pull request opened in your repository. Review it when you get a chance.",
        "Security code: 847291. This code expires in 10 minutes. Do not share with anyone.",
        "Your donation of $50 to Red Cross was processed. Thank you for your generosity!",
        "Your resume has been viewed by 3 recruiters this week on LinkedIn.",
        "I'll call you after the meeting to discuss the project updates.",
        "Can you pick up some groceries on your way home? We need milk and bread.",
        "No problem, take your time. The deadline isn't until next Friday.",
        "The conference call is at 2pm. Dial-in details are in the calendar invite.",
        "I've attached the revised budget spreadsheet. Please review when you can.",
        "Great news - the client approved the proposal! We can start next week.",
        "Remember to submit your expense reports before the end of the month.",
    ]

    rows = [(t, "spam") for t in spam] + [(t, "ham") for t in ham]
    df = pd.DataFrame(rows, columns=["text", "label"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─── Train ────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    print(f"\n Dataset: {len(df)} samples | "
          f"Spam: {(df.label == 'spam').sum()} | Ham: {(df.label == 'ham').sum()}")

    df["clean"] = df["text"].apply(preprocess_text)
    X, y = df["clean"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=8000,
            min_df=1, max_df=0.95, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=5.0, max_iter=500, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )),
    ])

    print("Running 5-fold cross-validation...")
    cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"Cross-val F1: {cv.mean():.4f} +/- {cv.std():.4f}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index("spam")]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, pos_label="spam")
    auc = roc_auc_score((y_test == "spam").astype(int), y_prob)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Spam F1 Score : {f1:.4f}")
    print(f"ROC-AUC       : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/spam_classifier.pkl")

    metrics = {
        "accuracy":   round(acc * 100, 2),
        "f1_score":   round(f1  * 100, 2),
        "roc_auc":    round(auc * 100, 2),
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "spam_count": int((df.label == "spam").sum()),
        "ham_count":  int((df.label == "ham").sum()),
        "cv_f1_mean": round(float(cv.mean()) * 100, 2),
        "cv_f1_std":  round(float(cv.std())  * 100, 2),
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved -> model/spam_classifier.pkl")
    print("Metrics saved -> model/metrics.json")
    return pipeline


if __name__ == "__main__":
    print("=" * 55)
    print("  Email Spam Detection - Model Training")
    print("=" * 55)
    df = build_dataset()
    train(df)
    print("\nDone! Run: python app.py")
