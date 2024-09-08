import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imutils
from pygame import mixer

# Initialize the mixer module to play sound alerts.
mixer.init()
# Load an alert sound file to be played during drowsiness detection.
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    # Calculate the eye aspect ratio (EAR) to determine blink or eye closure.
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    # Calculate the mouth aspect ratio (MAR) to detect yawning.
    A = distance.euclidean(mouth[3], mouth[9])  # Index 51, 59
    B = distance.euclidean(mouth[0], mouth[6])  # Index 48, 54
    mar = A / B
    return mar

# Set the thresholds for detecting eye closures and yawning.
thresh = 0.25  # Eye aspect ratio threshold for blink detection.
frame_check = 150  # Number of consecutive frames the eye must be below threshold.
mouth_thresh = 0.6  # Mouth aspect ratio threshold for yawning.
mouth_frame_check = 20  # Number of consecutive frames the mouth ratio must be exceeded.

# Initialize dlib's face detector and facial landmark predictor.
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Facial landmark indices for the left and right eye, and the mouth.
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)
flag = 0  # Counter for frames with eye closed.
yawn_flag = 0  # Counter for frames with yawning.

while True:
    ret, frame = cap.read()
    # Resize the frame for consistent processing speed.
    frame = imutils.resize(frame, width=450)
    # Convert the frame to grayscale to simplify face detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)  # Detect faces in the grayscale frame.

    for subject in subjects:
        shape = predict(gray, subject)  # Predict facial landmarks.
        shape = face_utils.shape_to_np(shape)  # Convert the landmarks to numpy array.

        # Extract eye and mouth coordinates from the landmarks.
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # Draw contours around the eyes and mouth for visualization.
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Check if the user's eyes have been closed for a prolonged period.
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                # Display an alert message on the frame.
                cv2.putText(frame, "ALERT: EYES CLOSED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()  # Play the alert sound.
        else:
            flag = 0  # Reset the frame counter if the eyes are not closed.

        # Check if the user is yawning.
        if mar > mouth_thresh:
            yawn_flag += 1
            if yawn_flag >= mouth_frame_check:
                # Display a yawning alert message.
                cv2.putText(frame, "YAWNING DETECTED", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            yawn_flag = 0  # Reset the yawning frame counter.

    # Display the frame with annotated features and alerts.
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Exit the loop if 'q' is pressed.
        break

# Cleanup: close the window, release the webcam, and stop the music.
cv2.destroyAllWindows()
cap.release()