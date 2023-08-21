from deepface import DeepFace
import os
import cv2


mode = "rtmp"

if mode == "file":
	base_folder = '/Users/taylorbrandl/Taylor/Python/Nimbus/DroneFollower/DALLE_emots/FERatt/FER2013Upscale/'

	# Load and preprocess the images
	num_images = 28#717
	correct_labels = []
	errors = 0
	i = 0
	# Iterate over subfolders in the base directory
	for emotion_folder in os.listdir(base_folder):
		emotion_path = os.path.join(base_folder, emotion_folder)
	
		if os.path.isdir(emotion_path):
			print(f"Processing images for emotion: {emotion_folder}")
			image_count = 0
	
			for imagefile in os.listdir(emotion_path):
				if imagefile.endswith(".jpg"):
					image_path = os.path.join(emotion_path, imagefile)
					try:
						image = cv2.imread(image_path)
						deepface_predict = DeepFace.analyze(image, actions=['emotion'])
						deepface_emotion = deepface_predict[0]['dominant_emotion']
						print(deepface_emotion)
						if deepface_emotion == emotion_folder:
							correct_labels.append(1)
						else:
							correct_labels.append(0)
					except:
						print("error")
						errors += 1

	
					image_count += 1
					i += 1
					print(f"Processed {image_count} images for emotion: {emotion_folder} count {i} of {num_images}")

					if i >= num_images:
						break
		if i >= num_images:
			break
		

	accuracy = sum(correct_labels)/len(correct_labels)
	errors = errors/num_images
	print(f"model accuracy:{accuracy}, error precentage:{errors}")

elif mode == "webcam":
	# Open webcam
	cap = cv2.VideoCapture(0)

	# Check if the webcam is opened correctly
	if not cap.isOpened():
		raise IOError("Cannot open webcam")

	while True:
		#ret, frame = cap.read()
		frame = cv2.imread('/Users/taylorbrandl/Downloads/webcam_Test.png')
		#frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
		deepface_predict = DeepFace.analyze(frame, actions=['region','emotion'], enforce_detection=False)
		deepface_emotion = deepface_predict[0]['dominant_emotion']
		rectangle = deepface_predict[0]['region']
		x = rectangle['x']
		y = rectangle['y']
		w = rectangle['w']
		h = rectangle['h']
		#print(deepface_emotion)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, str(deepface_emotion), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
		cv2.imshow('Input', frame)

		c = cv2.waitKey(1)
		if c == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

elif mode == "rtmp":
	# RTMP or RTMPS URL
	url = "rtmp://192.168.1.71/live/gopro"

	cap = cv2.VideoCapture(url)

	while True:
		ret, frame = cap.read()
		if not ret:
			print("Failed to grab frame")
			break

		deepface_predict = DeepFace.analyze(frame, actions=['region','emotion'], enforce_detection=False, silent=True)
		deepface_emotion = deepface_predict[0]['dominant_emotion']
		rectangle = deepface_predict[0]['region']
		x = rectangle['x']
		y = rectangle['y']
		w = rectangle['w']
		h = rectangle['h']
		#print(deepface_emotion)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, str(deepface_emotion), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

		cv2.imshow("RTMP Stream", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()