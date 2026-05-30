from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load vectorizer
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction page
@app.route('/predict', methods=['POST'])
def predict():

    # Get email text from form
    email = request.form['email']

    # Convert to list
    data = [email]

    # Transform text
    vector = cv.transform(data)

    # Predict
    prediction = model.predict(vector)

    # Result
    if prediction[0] == 1:
        result = "Spam Email"
    else:
        result = "Not Spam"

    return render_template(
        'index.html',
        prediction_text=result
    )


if __name__ == '__main__':
    app.run(debug=True)