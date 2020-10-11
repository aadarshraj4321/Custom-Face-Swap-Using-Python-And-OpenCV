
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


# create a zeros array canvas exactly in same size of "source_image_gray"
source_image_canvas = np.zeros_like(source_image_gray)

# get the shape of destination_image
height, width, no_of_channels = destination_image.shape

# create a zeros array canvas with size (height, width, no_of_channels)
destination_image_canvas = np.zeros_like((height,width,no_of_channels),np.uint8)



# Find faces in source image
# Return a numpy array that contains a histogram of the pixels in the image
source_faces = frontal_face_detector(source_image_gray)

# loop through all faces found in source image
for source_face in source_faces:
    # predictor takes human face (source_face) as input and is expected to itentify the
    # facial landmarks such as the corners of the mouth and eyes, tip of the nose , etc
    source_face_landmarks = frontal_face_predictor(source_image_gray,source_face)
    source_face_landmarks_points = []

    # loop through all 68 landmark points and add them to the tuple
    for landmark_no in range(0,68):
        x_point = source_face_landmarks.part(landmark_no).x
        y_point = source_face_landmarks.part(landmark_no).y
        source_face_landmarks_points.append((x_point,y_point))\

        #cv2.circle(source_image,(x_point,y_point),2,(255,0,0,),-1)
        #cv2.imshow("1 : landmark points of source", source_image)

    # convert to a integer base numpy array
    source_face_landmarks_points_array = np.array(source_face_landmarks_points, np.int32)

    # finds the convex hull contain indices of the contour points that make the hull.
    source_face_convexhull = cv2.convexHull(source_face_landmarks_points_array)

    #cv2.polylines(source_image,[source_face_convexhull],True,(255,0,0),3)
    #cv2.imshow("2: convex hull of source", source_image)

    # draw the filled polygon over the zero array canvas
    cv2.fillConvexPoly(source_image_canvas,source_face_convexhull,255)
    #cv2.imshow("3: create a canvas with of mask source", source_image_canvas)
    #cv2.waitKey(0)

    # place over the source image canvas
    source_face_image = cv2.bitwise_and(source_image,source_image,mask=source_image_canvas)
    cv2.imshow("4: Join canvas and the source image",source_face_image)
    cv2.waitKey(0)



    #print(source_face_landmarks_points)