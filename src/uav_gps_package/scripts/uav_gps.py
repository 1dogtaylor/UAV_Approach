#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import cv2
import numpy as np
from pyquaternion import Quaternion
from pynput import keyboard
import emotion_recognition

global_z = 0
global_x = 0
global_y = 0
global_quat = Quaternion(0,0,0,0)
stop_flag = False
emot_capture = False


def local_position_callback(data):
    global global_z, global_x, global_y, global_quat
    global_z = data.pose.pose.position.z
    global_x = data.pose.pose.position.x
    global_y = data.pose.pose.position.y
    global_quat = data.pose.pose.orientation
    #print(f"XYZQ: ({global_x}, {global_y}, {global_z}, {global_quat})")

def on_press(key):
	global stop_flag
	if key == keyboard.KeyCode(char='s'):
		print('tarting')
		stop_flag = True
		return False	#return false to stop the listener

def posupdate(center, shape, altitude, distance_behind):

    camera_pitch = 30 # angle of camera from horizon in degrees
    camera_fovV = 85 # vertical field of view in degrees
    camera_fovH = 85 # horizontal field of view in degrees

    # Add the target's pixel coordinates in the image frame
    target_pixel_x = center[0]  # X pixel coordinate of the target in the image frame (width)
    target_pixel_y = center[1]  # Y pixel coordinate of the target in the image frame (height)
    image_width = shape[1]  # Width of the image frame
    image_height = shape[0]  # Height of the image frame
   
    # Calculate the angle offsets in horizontal and vertical FOV for the target
    horizontal_angle_offset = (image_width / 2 - target_pixel_x) * (camera_fovH / image_width)
    vertical_angle_offset = (target_pixel_y - image_height / 2) * (camera_fovV / image_height)

    # Update the camera_pitch and heading to account for the target's position within the camera view
    camera_pitch += vertical_angle_offset

    # convert to radians
    camera_pitch = np.radians(camera_pitch)
    horizontal_angle_offset = np.radians(horizontal_angle_offset)
    
	# calculate the distance to the target
    target_distance_y = -(altitude / np.tan(camera_pitch))
    target_distance_x = np.abs(target_distance_y) * np.tan(horizontal_angle_offset)
    
    return target_distance_x, target_distance_y

def calculate_setpoint(target_distance_x, target_distance_y):
    global global_x, global_y


    setpoint_x = global_x + target_distance_x
    setpoint_y = global_y + target_distance_y

    return setpoint_x, setpoint_y


def main():
    # Initialize the ROS node and subscribe to the GPS data
	rospy.init_node('uav_gps_node', anonymous=True)
	rospy.Subscriber("/mavros/global_position/local", Odometry, local_position_callback)

    # Create a publisher for the setpoint_position/global topic
	setpoint_position_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

    # Set the fixed distance behind the target
	fixed_distance_behind_target = 6  # meters
	
	listener = keyboard.Listener(on_press=on_press)
	listener.start() #start the keyboard listener
	
	print("press s to start")
	


	while True: #wait for lower case s
		#print(f"XYZQ: ({global_x}, {global_y}, {global_z}, {global_quat})")
		if stop_flag:
			break
	

	# activate webcam
	webcam_index = 0
	video = cv2.VideoCapture(webcam_index)

	# Read the first frame
	ret, frame = video.read()

	# Select the object region in the first frame
	object_region = cv2.selectROI("Select Object", frame, False, False)

	# Initialize the tracker
	tracker = cv2.legacy.TrackerCSRT_create()
	tracker.init(frame, object_region)

	i = 0
	first_update = True
	moving = False
	z_points = []
	
	# uav is moved to the target altitude then the average altitude is found
	print('finding ave alt')
	j = 0
	while j < 500:
		z_points.append(global_z)
		j = j+1
		
	sum = 0
	for z in z_points:
		sum = sum+z
		
	static_z = sum/500
	
	print(f"ave alt is {static_z}")

	while not rospy.is_shutdown():

		
		ret, frame = video.read()
		shape = frame.shape

		if not moving:
			# Update the tracker
			ret, object_region = tracker.update(frame)

			# Draw the tracked object region and center
			if ret:
				p1 = (int(object_region[0]), int(object_region[1]))
				p2 = (int(object_region[0] + object_region[2]), int(object_region[1] + object_region[3]))
				center = (int(object_region[0] + object_region[2] / 2), int(object_region[1] + object_region[3] / 2))
				cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)

				# get the target's emotion
				if emot_capture:
					emot, rec_x, rec_y, rec_w, rec_h = emotion_recognition.emot_predict(frame)
					cv2.rectangle(frame, (rec_x, rec_y), (rec_x+rec_w, rec_y+rec_h), (0, 255, 0), 2)
					cv2.putText(frame, str(emot), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
				

				# Calculate the target's distance from the UAV
				target_distance_x, target_distance_y = posupdate(center, shape, static_z, fixed_distance_behind_target)
                
				# Calculate the setpoint position
				setpoint_x, setpoint_y = calculate_setpoint(target_distance_x, target_distance_y)
				if i%20==0: #print every 10 frames
					print(f"Target Distance: ({target_distance_x}, {target_distance_y})")
					print(f"Setpoint: ({setpoint_x}, {setpoint_y}) Update in {100-i} frames")
					print(f"XYZQ: ({global_x}, {global_y}, {global_z} )")
				#update setpoint every 100 frames
				if i >100:
					i = 0
					# check if target is within moving range
					if (np.sqrt(np.square(setpoint_x - global_x) + np.square(setpoint_y - global_y)) > 0.5 and (np.sqrt(np.square(setpoint_x - global_x) + np.square(setpoint_y - global_y))) < 5) or first_update:
						# Create a PoseStamped message
						setpoint_position = PoseStamped()
						setpoint_position.header.stamp = rospy.Time.now()
						setpoint_position.header.frame_id = "map"
						setpoint_position.pose.position = Point(setpoint_x, setpoint_y+fixed_distance_behind_target, static_z)
						setpoint_position.pose.orientation = global_quat

						setpoint_position_pub.publish(setpoint_position)

						print(f"\n\n---Position Updated: ({setpoint_x}, {setpoint_y+fixed_distance_behind_target}, {global_z}, {global_quat})\n\n")

						first_update = False

					else:
						print("Major or Minor change in target position")
				i+= 1

				# Display the center coordinates as text on the frame
				distance_to_target = np.round(np.sqrt(np.square(setpoint_x - global_x) + np.square(setpoint_y - global_y)), decimals=3)
				text = f"({np.round(target_distance_x, decimals=3)}, {np.round(target_distance_y, decimals=3)})"
				alt_text = f"alt {np.round(static_z,decimals=3)} m    delta {distance_to_target} m    {100-i}"
				cv2.putText(frame, text, (center[0] - 50 , center[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.625, (0, 0, 255), 2)#210
				cv2.putText(frame, alt_text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.625, (0,0,255), 2) 


			# Display the frame
			cv2.imshow("Frame", frame)

			# keyboard interupt
			key = cv2.waitKeyEx(1)
			if key == 113: #ASCII value for q
				break
			

	# Release resources and close windows
	video.release()
	cv2.destroyAllWindows()

	quit()


if __name__ == "__main__":
	main()

                

                
