import joblib
from flask import Flask, request, render_template

app = Flask(__name__, static_folder='statics')

artifact = joblib.load('model.joblib')
pipeline     = artifact['pipeline']
target_names = artifact['target_names']


@app.route('/', methods=['GET'])
def home():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '').strip()
    if not text:
        return render_template('predict.html', error='Please enter some text.')
    predicted_index = pipeline.predict([text])[0]
    predicted_label = target_names[predicted_index]
    return render_template('predict.html', result=predicted_label, input_text=text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
