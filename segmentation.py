import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
model = YOLO('best2.pt') 

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file.stream)
    
    # Run Inference
    results = model(img)[0]
    
    # dictionary to hold your desired format: {"ClassName": total_pixels}
    predictions = {}
    
    if results.masks is not None:
        # Move to CPU and convert to numpy for pixel counting
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        for i, mask in enumerate(masks):
            class_name = results.names[int(classes[i])]
            pixel_count = int(np.sum(mask > 0))
            
            # Aggregate the sum for each class
            predictions[class_name] = predictions.get(class_name, 0) + pixel_count

    # Process the result image (Clean: no boxes/labels)
    res_plotted = results.plot(labels=False, boxes=False)
    res_img = Image.fromarray(res_plotted[:, :, ::-1])

    # Encode Image to Base64
    buffered = io.BytesIO()
    res_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "status": "success",
        "predictions": predictions,
        "image": img_str
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)