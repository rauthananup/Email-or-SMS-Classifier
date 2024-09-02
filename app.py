from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your pre-trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    vect = vectorizer.transform(data)
    prediction = model.predict(vect)
    result = "Spam" if prediction[0] else "Ham"
    return render_template('index.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)

