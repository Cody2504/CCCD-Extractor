from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from modules.fullPipeline import FullPipelineOCR
from pyngrok import ngrok, conf

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = "static"

@app.route('/', methods = ['GET'])
def index():
    try:
        return render_template('formExtractor.html')
    except Exception as e:
        return f"Lỗi: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        image = request.files['file']
        
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(path_to_save)
        ocr_pipeline = FullPipelineOCR(model_path_4Corner="./weight/detectCorner.pt", model_path_textDet="./weight/detectText.pt")
        result = ocr_pipeline(path_to_save)
        if not (result is None): print(result)
        else: print("hehe") 
        os.remove(path_to_save)
        formatted_result = {
            "ID": result.get(0, ""),
            "Name": result.get(1, ""),
            "Date of Birth": result.get(2, ""),
            "Sex": result.get(3, ""),
            "Nationality": result.get(4, ""),
            "Place of Origin": result.get(5, ""),
            "Place of Residence": result.get(6, ""),
            "Expiry": result.get(7, "")
        }
        
        return jsonify({"result": formatted_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_ngrok():
    try:
        public_url = ngrok.connect(port=5000, proto='http', bind_tls=True)
        print(f"Public URL: {public_url}")
    except Exception as e:
        print(f"Error")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)