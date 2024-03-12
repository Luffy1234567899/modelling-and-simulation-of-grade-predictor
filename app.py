from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

data = pd.read_excel('student_performance_data.xlsx')
model = joblib.load('trained_linear_regression_model.pkl')

# Define your routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        prev_exam_grade = float(request.form['prev_exam_grade'])
        study_hours = float(request.form['study_hours'])
        
        # Prepare the input for prediction
        user_input = pd.DataFrame({'Study_Hours': [study_hours], 
                                   'Previous_Exam_Grades': [prev_exam_grade]})

        # Use the loaded model to predict
        predicted_grade = model.predict(user_input)[0]
        predicted_grade = min(predicted_grade, 100)

        y_true = data['Final_Grade']
        X_test = data[['Study_Hours', 'Previous_Exam_Grades']]
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r_squared = r2_score(y_true, y_pred)

        return render_template('result.html', predicted_grade=predicted_grade, mae=mae, mse=mse, r_squared=r_squared)

if __name__ == '__main__':
    app.run(debug=True)
