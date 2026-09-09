import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASETS
# ==========================================

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")


# ==========================================
# 2. ADD LABELS
# ==========================================

# 0 = Fake News
# 1 = Real News

fake["label"] = 0
true["label"] = 1


# ==========================================
# 3. CREATE CONTENT COLUMN
# ==========================================

# Combine title and article text into one field.
# fillna() handles missing values safely.

fake["content"] = (
    fake["title"].fillna("").astype(str)
    + " "
    + fake["text"].fillna("").astype(str)
)

true["content"] = (
    true["title"].fillna("").astype(str)
    + " "
    + true["text"].fillna("").astype(str)
)


# ==========================================
# 4. COMBINE DATASETS
# ==========================================

# Keep only the columns required for training.

data = pd.concat(
    [
        fake[["content", "label"]],
        true[["content", "label"]]
    ],
    ignore_index=True
)


# ==========================================
# 5. SHUFFLE DATA
# ==========================================

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 6. FEATURES AND TARGET
# ==========================================

X = data["content"]
y = data["label"]


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

# IMPORTANT:
# Split the raw text BEFORE fitting TF-IDF.
#
# This prevents information from the test set
# from being used when learning the vocabulary
# and IDF values.

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)


# Fit the vectorizer ONLY on training data.
X_train = vectorizer.fit_transform(X_train_text)

# Transform test data using the fitted vectorizer.
X_test = vectorizer.transform(X_test_text)


# ==========================================
# 9. CREATE LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    random_state=42
)


# ==========================================
# 10. TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# 11. MAKE PREDICTIONS
# ==========================================

pred = model.predict(X_test)


# ==========================================
# 12. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(
    y_test,
    pred
)

print("\n==========================================")
print("       FAKE NEWS DETECTION MODEL")
print("==========================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        pred,
        target_names=["Fake News", "Real News"]
    )
)


# ==========================================
# 13. SAVE TRAINED MODEL
# ==========================================

with open("fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)


# ==========================================
# 14. SAVE TF-IDF VECTORIZER
# ==========================================

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("==========================================")
print("Model saved successfully!")
print("Model file      : fake_news_model.pkl")
print("Vectorizer file : vectorizer.pkl")
print("==========================================")
