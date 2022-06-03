def detect_realtime():
    import cv2

    # Face classifier
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

    # Grab Webcam feed
    webcam = cv2.VideoCapture(0)

    # Show the current frame
    # to get a real-time stream, needs to put it in loop
    while True:
        # read current frame from webcam video stream
        successful_frame_read, frame = webcam.read() 

        # If there's an error, abort
        if not successful_frame_read:
            break
    
        # Change to greyscale, optimization
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces first 
        faces = face_detector.detectMultiScale(frame_grayscale)
    
        # print(faces) # To view the arrays of matrices

        # Run face detection within each of those faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face # 100, 200, 50
            cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50), 4)  
        

        # Get the sub-frame (the face only)
        # using numpy N-dimensional array slicing
        the_face = frame[y:y+h, x:x+w]
        
        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detect smiles
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors=7)

        #  # Find all smiles in the face
        # for (x_, y_, w_, h_) in smiles:
        #     # draw rectangle around the smiles
        #     cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 4)  # the values is BGR color code, 4 is thickness of rectangle
        
        # Label this face as smiling or not smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+30), fontScale=0.5,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
        else:
            cv2.putText(frame, 'Not smiling', (x, y+h+30), fontScale=0.5,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))     

        
        # Show current frame
        cv2.imshow('Smile Detector', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # Cleanup
    webcam.release() # exit webcam
    cv2.destroyAllWindows() # close all windows
    print("Code completed")

    return
