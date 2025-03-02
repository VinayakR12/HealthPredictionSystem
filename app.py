from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

with open("saved_models/DiabetesBestModel.pkl", "rb") as diabetes_model_file:
    diabetes_model = pickle.load(diabetes_model_file)

with open("saved_models/HeartBestModel.pkl", "rb") as heart_model_file:
    heart_model = pickle.load(heart_model_file)

with open("saved_models/ParkinsonsBestModel.pkl", "rb") as parkinsons_model_file:
    parkinsons_model = pickle.load(parkinsons_model_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html"), 404

@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    prediction = None
    if request.method == "POST":
        features = [float(request.form[key]) for key in [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]]

        result = diabetes_model.predict([features])[0]
        prediction = "Diabetic" if result == 1 else "Non-Diabetic"

    return render_template("diabetesForm.html", prediction=prediction)


@app.route("/heart", methods=["GET", "POST"])
def heart():
    prediction = None
    if request.method == "POST":

        user_input = [float(request.form[field]) for field in [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]]

        result = heart_model.predict([user_input])[0]
        prediction = "High Risk of Heart Disease" if result == 1 else "Low Risk of Heart Disease"

    return render_template("HeartForm.html", prediction=prediction)

@app.route("/parkinsons", methods=["GET", "POST"])
def parkinsons():
    prediction = None
    if request.method == "POST":

        features = [float(request.form[field]) for field in [
            "MDVP_Fo", "MDVP_Fhi", "MDVP_Flo", "MDVP_Jitter_percent", "MDVP_Jitter_Abs",
            "MDVP_RAP", "MDVP_PPQ", "Jitter_DDP", "MDVP_Shimmer", "MDVP_Shimmer_dB",
            "Shimmer_APQ3", "Shimmer_APQ5", "MDVP_APQ", "Shimmer_DDA", "NHR", "HNR",
            "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]]

        result = parkinsons_model.predict([features])[0]
        prediction = "Parkinson's Detected" if result == 1 else "No Parkinson's Detected"

    return render_template("parkinsonsForm.html", prediction=prediction)

#
# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    # Required for Render Deployment
    app.run(host="0.0.0.0", port=10000, debug=True)
