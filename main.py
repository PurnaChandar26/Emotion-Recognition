import cv2
import face_recognition
import os
import math
import numpy
from datetime import datetime
from deepface import DeepFace
import csv

allPaths = os.listdir("./class_data")
allNames = []
allRegNumbers = []
allEncodings = []
for index in range(len(allPaths)):
    allNames.append(allPaths[index].split(".")[0])
    allRegNumbers.append(allPaths[index].split(".")[1])
    image = face_recognition.load_image_file("./class_data/" + allPaths[index])
    temp = face_recognition.face_encodings(image)[0]
    allEncodings.append(temp)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")

prev_emotion = 'normal'
with open('./Attendance.csv', 'w') as file:
    file.write('time, regno, name, emotion\n')

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(
            imgS, actions=['emotion'], enforce_detection=False)
        # print(result['dominant_emotion'])

        facesInFrame = face_recognition.face_locations(imgS)

        encodeInFrame = face_recognition.face_encodings(imgS, facesInFrame)

        for endodeFaces, faceLoc in zip(encodeInFrame, facesInFrame):
            ismatched = face_recognition.compare_faces(
                allEncodings, endodeFaces)
            faceDis = face_recognition.face_distance(allEncodings, endodeFaces)
            try:
                bestMatchIndex = numpy.argmin(faceDis)
            except ValueError as ve:
                print(ve)
                continue

            if ismatched[bestMatchIndex]:
                matchedName = allNames[bestMatchIndex]

                current_emotion = result['dominant_emotion']
                if current_emotion != prev_emotion and current_emotion != 'neutral' and current_emotion != 'happy' and current_emotion != 'surprise':
                    # mark_inattentive(allRegNumbers[bestMatchIndex], matchedName, result['dominant_emotion'])
                    file.write(
                        f'{datetime.now()}, {allRegNumbers[bestMatchIndex]}, {matchedName}, {result["dominant_emotion"]}\n')
                    print(
                        f'{datetime.now()}, {allRegNumbers[bestMatchIndex]}, {matchedName}, {result["dominant_emotion"]}')
                    prev_emotion = result['dominant_emotion']

                x1, x2, y1, y2 = faceLoc
                x1, x2, y1, y2 = x1*4, x2*4, y1*4, y2*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, matchedName, (x1+6, y1-6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('video', img)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
