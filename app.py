from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from final import process_image_caption

app = Flask(__name__, static_url_path='/static')

# Set up paths
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'

# Set allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'jpg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        caption = process_image_caption(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('result.html', filename=filename, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
