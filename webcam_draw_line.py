"""
Live Pencil Drawing on Webcam Feed
-----------------------------------
Draw over your live video stream using the mouse, like a pencil.

Controls:
  Left-click + drag  -> draw
  1-5                -> change pencil color (red/green/blue/yellow/white)
  + / -               -> increase / decrease thickness
  c                   -> clear canvas
  s                   -> save current frame as image
  q                   -> quit

Requirements:
  pip install opencv-python numpy
"""

import cv2
import numpy as np

# ---- Global drawing state ----
drawing = False
prev_x, prev_y = -1, -1
color = (0, 0, 255)      # default: red (BGR)
thickness = 3
canvas = None            # persistent drawing layer


def draw(event, x, y, flags, param):
    global drawing, prev_x, prev_y, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (prev_x, prev_y), (x, y), color, thickness)
            prev_x, prev_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def main():
    global canvas, color, thickness

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Could not read from webcam.")
        return

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    window_name = "Live Pencil Drawing"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw)

    color_map = {
        ord('1'): (0, 0, 255),     # red
        ord('2'): (0, 255, 0),     # green
        ord('3'): (255, 0, 0),     # blue
        ord('4'): (0, 255, 255),   # yellow
        ord('5'): (255, 255, 255)  # white
    }

    print("Controls: 1-5 color | + / - thickness | c clear | s save | q quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view, feels natural

        # Merge canvas (drawing) onto the live frame
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        combined = cv2.add(frame_bg, canvas)

        # Small HUD showing current color/thickness
        cv2.circle(combined, (30, 30), thickness + 5, color, -1)
        cv2.putText(combined, "pencil", (55, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        elif key == ord('s'):
            cv2.imwrite("drawing_snapshot.png", combined)
            print("Saved drawing_snapshot.png")
        elif key in color_map:
            color = color_map[key]
        elif key == ord('+') or key == ord('='):
            thickness = min(thickness + 1, 20)
        elif key == ord('-') or key == ord('_'):
            thickness = max(thickness - 1, 1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
