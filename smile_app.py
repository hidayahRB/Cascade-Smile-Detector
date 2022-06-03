# import packages
import streamlit as st
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import cv2

def main():
    detector = MTCNN()
    st.title('WebApp Face Detector')
    st.header('Face Detection using Deep Learning')
    st.subheader('by Siti Norhidayah Abdul Bari Arbee')
    image_file = st.file_uploader('upload your image', type = ['png','jpg','jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        opencvImage = cv2.cvtColor(np.array(our_image),cv2.COLOR_RGB2BGR)
        st.text('Original image')
        st.image(our_image, width = 400)

    if st.button('draw boxes in image'):
        faces = detector.detect_faces(opencvImage)
        st.text(faces)
        image = ""
        for i in faces:
            x,y,width,height = i['box']
            image = cv2.rectangle(opencvImage, (x,y), (x+width, y+height), (255, 155, 55), 3)
            
        st.success('hey we found {} faces in image'.format(len(faces)))
        convertedImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(convertedImg, width = 400)
