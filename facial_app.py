import cv2
import streamlit as st
from PIL import Image
face_cascade = cv2.CascadeClassifier(
    'C:/Users/ass85/PycharmProjects/facial_app_exercise.py/.venv/Scripts/haarcascade_frontalface_default .xml')


def detect_faces(scaling, neighbors, color):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    continuons=True
    counter = 0
    while continuons==True:

        # Read the frames from the webcam
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a Pillow Image from the numpy array
        pil_image = Image.fromarray(rgb_frame)

        # Save the image as PNG

        pil_image.save(f"faces{counter}.png", "PNG")
        counter += 1

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaling, minNeighbors=neighbors)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)



        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continuons=False
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()



def hex_to_rgb(hex_color):
    # Vérifier si le code hexadécimal commence par "#"
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]

    # Extraction des composantes rouge, vert et bleu
    red = min(int(hex_color[0:2], 16), 255)
    green = min(int(hex_color[2:4], 16), 255)
    blue = min(int(hex_color[4:6], 16), 255)

    # Création du tuple (rouge, vert, bleu)
    rgb_tuple = (blue, green, red)

    return rgb_tuple



def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")

    color_choice = st.color_picker("Choose your color","#00FFAA")
    st.write(len(color_choice))
    col = hex_to_rgb(color_choice)

    st.write(col)

    scale = st.slider("Select your Scalefactor value", 1.01, 2.0, step=0.01, format="%.2f")
    st.write(scale)

    neighbor = st.slider("Select your MinNeighbors value", 1, 10)
    st.write(neighbor)

    u_save = st.button("Save images")

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
      detect_faces(scale, neighbor, col)



if __name__ == "__main__":
    app()

