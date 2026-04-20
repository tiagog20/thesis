import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("isOpened:", cap.isOpened())

ret, frame = cap.read()
print("ret:", ret)
print("frame is None:", frame is None)

if ret and frame is not None:
    cv2.imshow("Test Cam", frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()