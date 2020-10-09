
## Face Swapping

# import required libraries
import cv2
import dlib
import numpy as np

# initialize dlib library's face detector
# create dlib library's facila landmark predictor.
frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks.dat")

# read the source face image and convert it to grayscale
source_image = cv2.imread("images/jason.jpg")
source_image_copy = source_image
source_image_gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)

# read the destination face image and convert it to grayscale
destination_image = cv2.imread("images/brucewills.jpg")
destination_image_copy = destination_image
destination_image_gray = cv2.cvtColor(destination_image,cv2.COLOR_BGR2GRAY)
