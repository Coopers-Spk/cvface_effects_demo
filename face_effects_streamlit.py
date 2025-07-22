pip install opencv-python mediapipe numpy
import cv2
   

# Load OpenCV built-in pretained face dect (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

#Start webcam (0 = default, 1= external camera)
cap = cv2.VideoCapture(0)

mode = "face Detection"

def pixelated_face(frame, x1, y1, x2, y2):
    face_roi = frame[y1:y2, x1:x2] 
    small = cv2.resize(face_roi, (32, 32), interpolation=cv2.INTER_LINEAR)
    pixelated_face = cv2.resize(small, (face_roi.shape[1], face_roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = pixelated_face

def blurred_face(frame, x1, y1, x2, y2):
    face_roi = frame[y1:y2, x1:x2]
    blurred_face = cv2.GaussianBlur(face_roi, (55,55),30)
    frame[y1:y2, x1:x2] = blurred_face

def eye_smile_dector(gray, x, y, w, h):
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]
    roi_gray_lower = roi_gray[int(h/2):,:]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(
            roi_color,
            (ex, ey),
            (ex + ew, ey + eh),
            (0, 0, 255), 2
            )

    smiles = smile_cascade.detectMultiScale(roi_gray_lower, 1.7, 22)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(
            roi_color,
            (sx, sy + int(h/2)),
            (sx + sw, sy + sh + int(h/2)),
            (0, 0, 255), 2
            )


while True:
    #Captures frame from webcam (ret returns = true/fasle)
    ret, frame = cap.read()
    if not ret:
        break

    #Convert to Grey Scale & detects faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, #Haar Cascade is grey trained
        scaleFactor=1.3, #detects different sized facers 1.05 -1.2 better for small faces, 1.3-6 faster miss small faces
        minNeighbors=5, #<5 more faces & faulse positives, >5 less false pos but miss faces
        minSize=(30,30) #ignores faces < 30x30 pixels
        ) 

    #Draw box around detected faces + 20 pixels padding
    for (x,y,w,h) in faces:
        padding = 30
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, frame.shape[1])
        y2 = min(y + h + padding, frame.shape[0])

        if mode == "pixelated_face":
            pixelated_face(frame,x1, y1, x2, y2)
    
        elif mode == "blurred_face":
            blurred_face(frame, x1, y1, x2, y2)

        elif mode == "Eye & Smile Detection":
            eye_smile_dector(frame, x, y, w, h)
    
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255, 0), 2)
        
    cv2.imshow("Face Effect", frame)

    key = cv2.waitKey(1) & 0xFF
    #'q' to quit
    if key == ord('q'):
        break

    elif key == ord('p'):
        mode = "pixelated_face"

    elif key == ord('b'):
        mode = "blurred_face"

    elif key == ord('f'):
        mode = "face_Detection"

    elif key == ord('e'):
        mode = "Eye & Smile Detection"

cap.release()
cv2.destroyAllWindows()
