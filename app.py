from flask import Flask,request,jsonify,render_template
from gevent.pywsgi import WSGIServer
import numpy as np
import pandas as pd
import keras
import cv2
import tensorflow as tf
import os

app = Flask(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# model = tf.keras.layers.TFSMLayer("Notebook//BestModel.weights.h5", call_endpoint="serving_default")
model = keras.models.load_model("models//BestModel.keras")

IMG_SIZE = 224
MAX_SEQ_LENGTH = 100
NUM_FEATURES =2048
class_vocab = ['FAKE','REAL']

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling = "avg",
        input_shape = (IMG_SIZE,IMG_SIZE,3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    
    inputs = keras.Input((IMG_SIZE,IMG_SIZE,3))
    preprocessed = preprocess_input(inputs)
    
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs,outputs,name='feature_extractor')

feature_extractor = build_feature_extractor()

def crop_center_square(frame):
    y,x = frame.shape[0:2]
    min_dim = min(y,x)
    start_x = (x//2) - (min_dim//2)
    start_y = (y//2) - (min_dim//2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]

def load_video(path,max_frames=0,resize=(IMG_SIZE,IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame,resize)
            frame = frame[:,:,[2,1,0]]
            frames.append(frame)
            
            if len(frames) ==max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

from IPython import embed
def prepare_single_video(frames):
    frames=frames[None,...]
    frame_mask = np.zeros(shape=(1,MAX_SEQ_LENGTH,),dtype='bool')
    frame_features = np.zeros(shape=(1,MAX_SEQ_LENGTH,NUM_FEATURES),dtype='float32')
    
    for i,batch in enumerate(frames):
        Video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH,Video_length)
        for j in range(length):
            frame_features[i,j,:] = feature_extractor.predict(batch[None,j,:])
        frame_mask[i,:length] =  1
    return frame_features,frame_mask

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():   
    if 'video' not in request.files:
        return jsonify({'error':'No video file provided'}),400
    
    video =request.files['video']
    video_path = os.path.join("uploads",video.filename)
    video.save(video_path)
    
    frames = load_video(video_path)
    frame_features,frame_mask = prepare_single_video(frames)
    
    probabilities= model.predict([frame_features,frame_mask])[0]
    print(probabilities)
    dictionary = {}
    for i in np.argsort(probabilities)[::-1]:
        dictionary[class_vocab[i]] = f'{probabilities[i]*100:5.2f}%'
    os.remove(video_path)
    print(dictionary)
    return jsonify({"Real":dictionary['REAL'],"Fake":dictionary['FAKE']})
    
    print(dictionary)
    # return jsonify({"result":result,'confidence':confidence})

if __name__ == '__main__':
    app.run()
