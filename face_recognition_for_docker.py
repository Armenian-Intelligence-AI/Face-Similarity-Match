from flask import Flask, request
import cv2
import numpy as np
import face_recognition
from numpy.linalg import norm
import base64
import json
import socket

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
app = Flask(__name__)



def similarity_checker(id_photo:str,ekeng_photo:str,threshold = 0.95):
    similarity_scores = []
    unkown_img_rotated_face_encodings  = []
    try:
        binary_data = base64.b64decode(id_photo)
        data = np.frombuffer(binary_data,dtype = np.uint8)
        unknown_image_0deg = cv2.imdecode(data,cv2.IMREAD_COLOR)
    except: 
        return False
        
    
    unknown_image_90deg = cv2.rotate(unknown_image_0deg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    unknown_image_180deg = cv2.rotate(unknown_image_0deg,cv2.ROTATE_180)
    unknown_image_270deg = cv2.rotate(unknown_image_0deg,cv2.ROTATE_90_CLOCKWISE)
    unkown_img_rotation_encodings = [unknown_image_0deg,unknown_image_90deg,unknown_image_180deg, unknown_image_270deg]
    
    for rotated_img in unkown_img_rotation_encodings:
        try:
            unkown_img_rotated_face_encodings.append(face_recognition.face_encodings(rotated_img)[0])
        except: continue
    
    if len(unkown_img_rotated_face_encodings) == 0:
        return False
    

        
    try:
        binary_data = base64.b64decode(ekeng_photo)
        data = np.frombuffer(binary_data,dtype = np.uint8)
        known_image = cv2.imdecode(data,cv2.IMREAD_COLOR)
        known_img_face_encodings = face_recognition.face_encodings(known_image)[0]
    except:
        return False
            
    for img in unkown_img_rotated_face_encodings:
        cosine_score  = np.dot(img,known_img_face_encodings)/(norm(img)*norm(known_img_face_encodings))
        similarity_scores.append(cosine_score)

    final_similarity = max(similarity_scores)
    
    if final_similarity >= threshold:
    	return True
    else:
    	return "Upload another image and try once again!"
    	
@app.route('/face_recognition', methods=['POST'])
def handle_face_recognition():
    # This function will be called when a POST request is made to /webhook
    # You can access the data sent in the request using request.data or request.json
    try:
        data = json.loads(request.json)
        # Handle the data as needed
        print("Received POST request with data:")
        print(data)

        # You can send a response back if required
        response = similarity_checker(data["uploaded_photo"],data["ekeng_photo"])
        return response, 200  # 200 OK status code
    except Exception as e:
        error_message = str(e)
        return error_message, 500
        
if __name__ == '__main__':
    # Run the Flask app on a specified host and port
    app.run(host=IPAddr, port=8080, debug=False)
