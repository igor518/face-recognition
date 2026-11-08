import os
import cv2
import numpy as np
from PIL import Image
from IPython.display import display
from ultralytics import YOLO


def main():
    folder = "faces/Ihor"
    files = os.listdir(folder)
    face_model = YOLO("yolov8m-face.pt")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    for file in files:
        photo = cv2.imread(folder + "/" + file)
        if photo is None:
            continue

        face_result = face_model(photo, verbose=False)
        if not face_result[0].boxes:
            continue

        left, top, right, bottom = face_result[0].boxes.xyxy[0].int()
        face = photo[top:bottom, left:right]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (200, 200))
        faces.append(face)
        labels.append(0)

    if not faces:
        raise RuntimeError(f"No faces were detected in {folder}")

    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write("face_recognizer.yml")



if __name__ == '__main__':
   main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
