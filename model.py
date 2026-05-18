import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Create content column
fake["content"] = (
    fake["title"].fillna("") + " " +
    fake["text"].fillna("")
)

true["content"] = (
    true["title"].fillna("") + " " +
    true["text"].fillna("")
)

# Combine datasets
data = pd.concat(
    [fake, true],
    ignore_index=True
)

# Shuffle data
data = data.sample(
    frac=1,
    random_state=42
)

# Reset index
data.reset_index(
    drop=True,
    inplace=True
)

# Features + target
X = data["content"]
y = data["label"]

# Convert text into vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

# Train
model.fit(
    X_train,
    y_train
)

# Predict
pred = model.predict(X_test)

# Accuracy
print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        pred
    )
)

# Save model
pickle.dump(
    model,
    open(
        "fake_news_model.pkl",
        "wb"
    )
)

# Save vectorizer
pickle.dump(
    vectorizer,
    open(
        "vectorizer.pkl",
        "wb"
    )
)

print("\nModel saved successfully")