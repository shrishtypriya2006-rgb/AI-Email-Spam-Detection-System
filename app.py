from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load CountVectorizer
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get email text
    email = request.form['email']

    # Convert into list
    data = [email]

    # Convert text into vector
    vector = cv.transform(data)

    # Predict spam or ham
    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "Spam Email"
    else:
        result = "Not Spam"

    return render_template(
        'index.html',
        prediction_text=result
    )

# Run app
if __name__ == '__main__':
    app.run(debug=True)