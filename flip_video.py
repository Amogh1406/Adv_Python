"""
Flip AND continuously rotate a video (webcam or .mp4 file) through 360 degrees.

Controls:
  h -> flip horizontal
  v -> flip vertical
  b -> flip both
  n -> no flip
  r -> toggle continuous 360-degree rotation on/off
  + -> rotate faster
  - -> rotate slower
  q -> quit

Requirements:
  pip install opencv-python
"""

import cv2

# Use 0 for webcam, or a filename e.g. "input.mp4"
SOURCE = 0  # change to 0 for webcam

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("Could not open video source.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps and fps > 0 else 30

flip_code = None      # None / 1 (horizontal) / 0 (vertical) / -1 (both)
rotating = False
angle = 0
rotate_speed = 3       # degrees per frame

print("Controls: h/v/b/n flip | r toggle 360 rotate | +/- speed | q quit")

while True:
    ret, frame = cap.read()
    if not ret:
        # loop the video file back to the start instead of quitting
        if SOURCE != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        break

    if flip_code is not None:
        frame = cv2.flip(frame, flip_code)

    if rotating:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame = cv2.warpAffine(frame, matrix, (w, h))
        angle = (angle + rotate_speed) % 360

    cv2.imshow("Flip + 360 Rotate", frame)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        flip_code = 1
    elif key == ord('v'):
        flip_code = 0
    elif key == ord('b'):
        flip_code = -1
    elif key == ord('n'):
        flip_code = None
    elif key == ord('r'):
        rotating = not rotating
    elif key in (ord('+'), ord('=')):
        rotate_speed = min(rotate_speed + 1, 30)
    elif key in (ord('-'), ord('_')):
        rotate_speed = max(rotate_speed - 1, 1)

cap.release()
cv2.destroyAllWindows()