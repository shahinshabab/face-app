import os
import cv2
import face_recognition
import streamlit as st
import numpy as np
import time
from PIL import Image

# Path to the database folder where face encodings will be saved
DATABASE_PATH = "user_faces"
if not os.path.exists(DATABASE_PATH):
    os.makedirs(DATABASE_PATH)

def add_new_user_face(name, image):
    # Convert the uploaded image to a numpy array
    img = np.array(image)
    # Find the face locations and encodings in the image
    face_encodings = face_recognition.face_encodings(img)
    
    if len(face_encodings) == 0:
        st.error("No face detected in the image. Please try again with a clearer image.")
        return
    
    # Check if the name already exists in the database
    encoding_file_path = os.path.join(DATABASE_PATH, f"{name}.npy")
    if os.path.exists(encoding_file_path):
        st.error(f"A face for {name} already exists in the database. Please choose a different name.")
        return
    
    # Save the encoding as a file
    face_encoding = face_encodings[0]
    np.save(encoding_file_path, face_encoding)
    st.success(f"Face for {name} added to the database!")


def recognize_face(image):
    # Convert the uploaded image to a numpy array
    img = np.array(image)
    
    # Find face locations and encodings in the uploaded image
    face_encodings = face_recognition.face_encodings(img)
    
    if len(face_encodings) == 0:
        st.error("No face detected in the image. Please try again with a clearer image.")
        return None
    
    # If multiple faces are detected, you can either ask the user to upload a clearer image or handle it in some way
    if len(face_encodings) > 1:
        st.warning("Multiple faces detected. Only the first face will be used for recognition.")
    
    # Compare the face encoding with existing encodings in the database
    for encoding_file in os.listdir(DATABASE_PATH):
        known_encoding = np.load(os.path.join(DATABASE_PATH, encoding_file))
        results = face_recognition.compare_faces([known_encoding], face_encodings[0])
        
        if results[0]:  # If a match is found
            name = encoding_file.split(".")[0]
            return name
    
    return None  # No match found


# Function to capture face from webcam (optional if using local capture)
def capture_face_from_webcam():
    st.write("Please position your face in front of the webcam for detection...")
    camera = cv2.VideoCapture(0)
    time.sleep(1)  # Give camera some time to start

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Convert frame to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Show the current frame
        st.image(frame, channels="BGR", use_column_width=True)

        if len(face_encodings) > 0:
            return frame, face_encodings[0]
        
        # Stop if the user presses 'q' or some condition to stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    return None, None  # No face detected

# Main user page logic
def user_page():
    st.title("Face Recognition System")

    # Allow the user to either add a new face or check if an existing user
    action = st.selectbox("Choose action:", ("Add New Face", "Recognize Face"))

    if action == "Add New Face":
    st.subheader("Add Your Face to the System")
    name = st.text_input("Enter your name:")
    uploaded_image = st.file_uploader("Upload an image of your face:", type=["jpg", "png", "jpeg"])
    use_webcam = st.button("Use Webcam to Capture Face")
    
    if use_webcam:
        frame, face_encoding = capture_face_from_webcam()
        if frame is not None:
            # You can process the captured frame as needed here
            encoding_file_path = os.path.join(DATABASE_PATH, f"{name}.npy")
            np.save(encoding_file_path, face_encoding)
            st.success(f"Face for {name} added to the database!")
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        if st.button("Add Face"):
            add_new_user_face(name, image)

