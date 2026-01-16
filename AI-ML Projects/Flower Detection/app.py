from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("iris_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     features = [float(x) for x in request.form.values()]
#     final_features = np.array([features])
#     prediction = model.predict(final_features)[0]

#     species = ["Setosa", "Versicolor", "Virginica"]
#     result = species[prediction]

#     return render_template("index.html", prediction_text=f"Predicted Species: {result}")
@app.route("/predict", methods=["POST"])
def predict():
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    final_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(final_features)[0]

    species = ["Setosa", "Versicolor", "Virginica"]
    result = species[prediction]

    return render_template("index.html", prediction_text=f"Predicted Species: {result}")

if __name__ == "__main__":
    app.run(debug=True)
