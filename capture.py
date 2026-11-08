import os
import cv2
from ultralytics import YOLO


def main():
    cap = cv2.VideoCapture(1)
    face_model = YOLO("yolov8m-face.pt")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("face_recognizer.yml")
    names = {0: "Ihor"}

    while cv2.waitKey(1) != ord("x"):
        _, frame = cap.read()
        face_result = face_model(frame, verbose=False)
        processed_feed = face_result[0].plot()

        left, top, right, bottom = face_result[0].boxes.xyxy[0].int()
        face = frame[top:bottom, left:right]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (200, 200))

        pred_label, distance = face_recognizer.predict(face)
        cv2.putText(
            processed_feed,
            str(names[pred_label]),
            (int(left), int(bottom) + 20),
            0,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("my window", processed_feed)



if __name__ == "__main__":
    main()