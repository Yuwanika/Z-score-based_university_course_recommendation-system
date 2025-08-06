# app.py

from flask import Flask,render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

# Load models and encoders
stage1_model = joblib.load('ml-models/stage1_model.pkl')
stage2_model = joblib.load('ml-models/stage2_model.pkl')
label_encoders = joblib.load('ml-models/label_encoders.pkl')
scaler = joblib.load('ml-models/scaler.pkl')
poly = joblib.load('ml-models/poly.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        z_score = float(data['z_score'])
        stream = data['stream']
        district = data['district']
        year = int(data['year'])

        stream_encoded = label_encoders['stream'].transform([stream])[0]
        district_encoded = label_encoders['district'].transform([district])[0]
        stream_district = f"{stream}|{district}"
        stream_district_encoded = label_encoders['stream_district'].transform([stream_district])[0]

        # Feature engineering (before scaling)
        zscore_poly = poly.transform([[z_score]])
        zscore_squared = zscore_poly[0][1]
        zscore_trend = 0
        zscore_district = z_score * district_encoded

        # Scale numerical features together
        numerical_input = [[z_score, zscore_district, zscore_squared, zscore_trend]]
        scaled_numerical = scaler.transform(numerical_input)
        z_score_scaled, zscore_district_scaled, zscore_squared_scaled, zscore_trend_scaled = scaled_numerical[0]

        # Create full input
        input_data = np.array([[
            z_score_scaled,
            stream_encoded,
            district_encoded,
            year,
            zscore_district_scaled,
            zscore_squared_scaled,
            zscore_trend_scaled,
            stream_district_encoded
        ]])

        # Stage 1: predict top 10
        stage1_probs = stage1_model.predict_proba(input_data)[0]
        top10_indices = np.argsort(stage1_probs)[-10:]
        top10_courses = label_encoders['university_course'].inverse_transform(top10_indices)
        top10_probs = stage1_probs[top10_indices]

        # Stage 2: rerank top 10
        stage2_inputs = [np.append(input_data[0], idx) for idx in top10_indices]
        stage2_probs = stage2_model.predict_proba(np.array(stage2_inputs))[:, 1]
        top1_index = top10_indices[np.argmax(stage2_probs)]
        top1_course = label_encoders['university_course'].inverse_transform([top1_index])[0]

        # Format results
        top10_results = [
            {'university': course.split('|')[0], 'course': course.split('|')[1], 'probability': float(prob)}
            for course, prob in zip(top10_courses, top10_probs)
        ]
        top1_result = {
            'university': top1_course.split('|')[0],
            'course': top1_course.split('|')[1]
        }

        return jsonify(top1_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
