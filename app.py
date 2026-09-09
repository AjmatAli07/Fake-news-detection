from flask import Flask, render_template, request
import pickle
from pathlib import Path


# ==========================================
# 1. CREATE FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# 2. DEFINE PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


# ==========================================
# 3. LOAD TRAINED MODEL AND VECTORIZER
# ==========================================

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    print("Model and vectorizer loaded successfully.")

except FileNotFoundError as error:
    model = None
    vectorizer = None

    print(f"Error: Required model file not found - {error}")


# ==========================================
# 4. HOME ROUTE
# ==========================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        result_type=None,
        news=""
    )


# ==========================================
# 5. PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether model and vectorizer are available
    if model is None or vectorizer is None:
        return render_template(
            "index.html",
            prediction="Model files are not available. Please train the model first.",
            result_type="error",
            news=""
        )

    try:
        # Get news text from the HTML form
        news = request.form.get("news", "").strip()

        # Check for empty input
        if not news:
            return render_template(
                "index.html",
                prediction="Please enter news text.",
                result_type="error",
                news=""
            )

        # ==========================================
        # 6. CONVERT NEWS TEXT INTO TF-IDF FEATURES
        # ==========================================

        data = vectorizer.transform([news])


        # ==========================================
        # 7. MAKE PREDICTION
        # ==========================================

        prediction = model.predict(data)


        # ==========================================
        # 8. CONVERT MODEL OUTPUT INTO RESULT
        # ==========================================

        if prediction[0] == 1:
            result = "🟢 Real News"
            result_type = "real"
        else:
            result = "🔴 Fake News"
            result_type = "fake"


        # ==========================================
        # 9. DISPLAY RESULT
        # ==========================================

        return render_template(
            "index.html",
            prediction=result,
            result_type=result_type,
            news=news
        )

    except Exception:
        return render_template(
            "index.html",
            prediction="Unable to analyze the news. Please try again.",
            result_type="error",
            news=news
        )


# ==========================================
# 10. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
