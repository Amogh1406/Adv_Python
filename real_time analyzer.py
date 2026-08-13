"""
Real-Time Image Analyzer (OpenCV)
----------------------------------
Captures webcam feed and analyzes it live: brightness/contrast, edges,
histogram, motion, grayscale. Overlay shows FPS + live stats.

Controls:
  1 - Original
  2 - Grayscale
  3 - Edge detection (Canny)
  4 - Histogram overlay
  5 - Motion detection (frame diff)
  s - Save current frame
  q - Quit
"""

import cv2
import numpy as np
import time

MODES = {
    ord('1'): "original",
    ord('2'): "grayscale",
    ord('3'): "edges",
    ord('4'): "histogram",
    ord('5'): "motion",
}


def draw_histogram(frame, width=256, height=200):
    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)
        for x in range(1, width):
            cv2.line(hist_img,
                      (x - 1, height - int(hist[x - 1])),
                      (x, height - int(hist[x])),
                      color, 1)
    return hist_img


def overlay_stats(frame, fps, brightness, contrast):
    lines = [f"FPS: {fps:.1f}", f"Brightness: {brightness:.1f}", f"Contrast: {contrast:.1f}"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    mode = "original"
    prev_gray = None
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Stats (always computed)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        # Mode-specific display
        if mode == "original":
            display = frame

        elif mode == "grayscale":
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif mode == "edges":
            edges = cv2.Canny(gray, 100, 200)
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif mode == "histogram":
            hist_img = draw_histogram(frame)
            h, w = frame.shape[:2]
            hist_resized = cv2.resize(hist_img, (w // 3, h // 3))
            display = frame.copy()
            display[10:10 + hist_resized.shape[0], w - 10 - hist_resized.shape[1]:w - 10] = hist_resized

        elif mode == "motion":
            if prev_gray is None:
                prev_gray = gray
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            prev_gray = gray

        else:
            display = frame

        display = overlay_stats(display, fps, brightness, contrast)
        cv2.putText(display, f"Mode: {mode}", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Real-Time Image Analyzer", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{int(time.time())}.png"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
        elif key in MODES:
            mode = MODES[key]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()