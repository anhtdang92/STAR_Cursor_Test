import numpy as np
import cv2

# Create a test video
def create_test_video(output_path, duration=5, fps=30, width=1280, height=720):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        # Create a frame with a moving pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = int((i / (duration * fps)) * width)
        cv2.rectangle(frame, (x, 0), (x + 100, height), (0, 255, 0), -1)
        cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()

if __name__ == '__main__':
    create_test_video('input/video/test.mp4') 