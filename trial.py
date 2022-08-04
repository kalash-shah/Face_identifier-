import cv2

frame_cap = cv2.VideoCapture(0)

while True:
    ret, frame = frame_cap.read()
    cv2.imshow('frame', frame)
    #for exiting the window
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break