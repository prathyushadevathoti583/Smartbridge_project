# Import the libraries
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
import tensorflow_hub as hub
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # ✅ Add this


model = load_model("model.h5", custom_objects={'KerasLayer': hub.KerasLayer})
# Replace with your model file name

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Home page route - renders index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route to details.html for uploading image
@app.route('/details')
def details():
    return render_template('details.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('details'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('details'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(96, 96))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return render_template("results.html", prediction=predicted_class, image_file=filename)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
