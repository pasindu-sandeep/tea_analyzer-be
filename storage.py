from flask import Flask, request, jsonify
from google.cloud import storage
import uuid

app = Flask(__name__)

PROJECT_ID = "speech-to-text-488803"
BUCKET_NAME = "tea-analyzer-image-bucket"

def upload_image(file):

    client = storage.Client(project=PROJECT_ID)

    bucket = client.bucket(BUCKET_NAME)

    filename = str(uuid.uuid4()) + "_" + file.filename

    blob = bucket.blob(filename)

    blob.upload_from_file(file)

    url = f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"

    return url

