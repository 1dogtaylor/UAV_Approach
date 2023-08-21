from deepface import DeepFace
import cv2


# Check if the webcam is opened correctly
def emot_predict(frame):
	deepface_predict = DeepFace.analyze(frame, actions=['region','emotion'], enforce_detection=False, silent=True)
	deepface_emotion = deepface_predict[0]['dominant_emotion']
	rectangle = deepface_predict[0]['region']
	x = rectangle['x']
	y = rectangle['y']
	w = rectangle['w']
	h = rectangle['h']
	#print(deepface_emotion)
	
	return deepface_emotion, x, y, w, h

