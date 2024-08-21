import keras
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
from pymongo import MongoClient

connection_string = 'mongodb+srv://sikandarnust1140:ZBXI5No3tsTeKb0u@cluster0.mo69b0z.mongodb.net/newDB?retryWrites=true&w=majority'
client = MongoClient(connection_string)
dataBase = client["Machine_Learning_Practice"]
user_collection = dataBase["userInfo"]
data_collection = dataBase['Heart_Disease_Patients_Data']

app = Flask(__name__)

try:
    model = keras.models.load_model("heart_disease_model.h5")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Model file not found. Ensure the path is correct.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


@app.route("/")
def index():
    return render_template("register.html")


@app.route("/register", methods=['POST'])
def register():
    data = request.form
    name = data.get("name")
    address = data.get('address')
    email = data.get('email')
    phone_no = data.get('phone_no')

    if not address or not name or not email or not phone_no:
        return {"error": "Missing data"}, 404

    user_data = {
        "name": name,
        "address": address,
        "phone_no": phone_no,
        "email": email
    }


    try:
        user_collection.insert_one(user_data)
        return render_template("index.html")
    except Exception as e:
        return {"error2": "SignUp Unsuccessful", "details": str(e)}, 500

@app.route("/main")
def main():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.form
        age = data.get("age")
        sex = data.get("sex")
        chest_pain_type = data.get("chest_pain_type")
        bp = data.get("bp")
        cholesterol = data.get("cholesterol")
        fbs_over_120 = data.get("fbs")
        ekg_results = data.get("ekg_results")
        max_hr = data.get("max_hr")
        exercise_angina = data.get("exercise_angina")
        st_depression = data.get("st_depression")
        slope_of_st = data.get("slope_of_st")
        num_vessels_fluro = data.get("num_vessels_fluro")
        thallium = data.get("thallium")
        input_data = [age, sex, chest_pain_type, bp, cholesterol, fbs_over_120, ekg_results, max_hr, exercise_angina,
                      st_depression, slope_of_st, num_vessels_fluro, thallium]

        if input_data is None:
            return jsonify({"error": "No input data provided"}), 400

        data_np = np.array(input_data).reshape(1, -1)
        prediction = model.predict(scaler.transform(data_np))

        result = "This patient has heart disease" if prediction >= 0.5 else "This patient does not have heart disease"
        print(result)
        # Store the patient's data along with the prediction result
        patient_data = {
            "age": age,
            "sex": sex,
            "chest_pain_type": chest_pain_type,
            "bp": bp,
            "cholesterol": cholesterol,
            "fbs_over_120": fbs_over_120,
            "ekg_results": ekg_results,
            "max_hr": max_hr,
            "exercise_angina": exercise_angina,
            "st_depression": st_depression,
            "slope_of_st": slope_of_st,
            "num_vessels_fluro": num_vessels_fluro,
            "thallium": thallium,
            "result" : result
        }



        data_collection.insert_one(patient_data)

        return render_template("result.html", result=result)
    except Exception as e:
        return jsonify({"error1": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)