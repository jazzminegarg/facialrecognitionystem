import numpy as np
import face_recognition
import cv2
import csv
from datetime import datetime

video_capture=cv2.VideoCapture(0)

# load known faces
sk_image=face_recognition.load_image_file("Faces/sk1.jpeg")
sk_encoding=face_recognition.face_encodings(sk_image)[0]
sid_image=face_recognition.load_image_file("Faces/sid.jpeg")
sid_encoding=face_recognition.face_encodings(sid_image)[0]
jas_image=face_recognition.load_image_file("Faces/jas.jpg")
jas_encoding=face_recognition.face_encodings(jas_image)[0]
ass_image=face_recognition.load_image_file("Faces/assi.png")
ass_encoding=face_recognition.face_encodings(ass_image)[0]

aar_image=face_recognition.load_image_file("Faces/aarushi.png")
aar_encoding=face_recognition.face_encodings(aar_image)[0]
known_face_encodings=[sk_encoding,sid_encoding,jas_encoding,ass_encoding,aar_encoding]

known_face_names=["sk","sid","jas","ass","aarushi"]

# list of expected students
students=known_face_names.copy()

face_locations=[]
face_encodings=[]

# get current date and time

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")
f=open(f"{current_date}.csv","w+",newline="")
lnwriter=csv.writer(f)

while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations=face_recognition.face_locations(rgb_small_frame)
    face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance=face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index=np.argmin(face_distance)
        if(matches[best_match_index]):
            name=known_face_names[best_match_index]

        # Add the text if a person is present
        if name in known_face_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale=1.5
            fontColor=(255,0,0)
            thickness=3
            linetype=2
            cv2.putText(frame,name+" chutiya hai",bottomLeftCornerOfText,font,fontScale,fontColor,thickness,linetype)

            if name in students:
                students.remove(name)
                current_time=now.strftime("%H-%M-%S")
                lnwriter.writerow(name)
    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()
