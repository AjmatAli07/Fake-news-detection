from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        news=""
    )

@app.route("/predict", methods=["POST"])
def predict():

    try:
        news = request.form["news"].strip()

        if news == "":
            return render_template(
                "index.html",
                prediction="Please enter news text",
                result_type="error",
                news=""
            )

        data = vectorizer.transform([news])

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "🟢 Real News"
            result_type = "real"
        else:
            result = "🔴 Fake News"
            result_type = "fake"

        return render_template(
            "index.html",
            prediction=result,
            result_type=result_type,
            news=news
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            result_type="error",
            news=""
        )

if __name__ == "__main__":
    app.run(debug=True)