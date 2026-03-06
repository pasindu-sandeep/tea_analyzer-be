import io
from flask import Flask, request, send_file, json
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
# Replace with the path to your trained segmentation model
model = YOLO('best2.pt') 

@app.route('/find_disease', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files['image']
    # Efficiently load image data into PIL directly
    img = Image.open(file.stream)
    
    # Run segmentation inference
    results = model(img)[0]
    
    # Create the prediction summary
    summary = [{"class": results.names[int(b.cls)], "conf": float(b.conf)} for b in results.boxes]

    # Plot results without labels (confidences) and without bounding boxes
    # The `labels=False` and `boxes=False` parameters disable these elements.
    res_plotted = results.plot(labels=False, boxes=False) 
    
    # Convert BGR (OpenCV format) back to RGB for PIL
    res_img = Image.fromarray(res_plotted[:, :, ::-1])
    
    # Save the plotted image data to an in-memory byte stream
    img_io = io.BytesIO()
    res_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)

    # Return the raw image file directly as the response
    response = send_file(img_io, mimetype='image/jpeg')
    
    # Store the prediction summary in a custom JSON header
    response.headers['X-Prediction-Summary'] = json.dumps(summary)
    return response

if __name__ == '__main__':
    # Adjust host and port if needed
    app.run(host='0.0.0.0', port=8080)