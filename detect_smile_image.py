# import package
import cv2
import numpy as np

# load file
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# get image
image = cv2.imread("test-image-3.jpg")

# Change to greyscale, optimization
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces first 
faces = face_detector.detectMultiScale(img_gray, 1.4, 6)

# Run face detection within each of those faces
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    # the values is BGR color code, 4 is thickness of rectangle
    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 155, 55), 3)

# Show current frame
# cv2.imshow('Detect Face First', image)

# Get the sub-frame (the face only)
# using numpy N-dimensional array slicing
the_face = image[y:y+h, x:x+w]

# Show current frame
# cv2.imshow('Cutted Face', the_face)

# Change to grayscale
face_gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

# Show current frame
# cv2.imshow('Convert Face To Grayscale', face_gray)

# Detect smiles
smiles = smile_detector.detectMultiScale(face_gray, scaleFactor = 1.7, minNeighbors=20)

# Label this face as smiling
if len(smiles) > 0:
    cv2.putText(image, 'Smiling', (x, y+h+30), fontScale=0.5,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
    # Show current frame
    cv2.imshow('Smiling', image)
else :
    cv2.putText(image, 'Not Smiling', (x, y+h+30), fontScale=0.5,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
    # Show current frame
    cv2.imshow('Not Smiling', image)

    
#cv2.imshow('Smile Detector', image)
cv2.waitKey(0)
cv2.destroyAllWindows()