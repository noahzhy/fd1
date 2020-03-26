import cv2
import time
import os
from collections import deque

def get_video(frames, save_path):

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 5
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_name = '{}.mp4'.format(timestamp)
    full_path = os.path.join(save_path, save_name)
    out = cv2.VideoWriter(full_path, fourcc, fps, (320, 240))

    if len(frames) == 50:
        # tmp_frames = frames
        for a in list(frames):
            out.write(a)

    out.release()
    return os.path.join(os.getcwd(), save_path, save_name)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    frames = deque(maxlen=50)
    get_video(frames, "output")