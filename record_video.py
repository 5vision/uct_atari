import cv2
import numpy as np
import cPickle
import argparse

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('fname', type=str, help="Filename with frames and actions.")
parser.add_argument('-o', '--output', type=str, default='output.avi',
                    help="Output filename.")
parser.add_argument('-f', '--fps', type=int, default=10,
                    help="Frames per second.")
parser.add_argument('-c', '--codec', type=str, default='XVID',
                    help="Codec for video.")
args = parser.parse_args()


with open(args.fname, 'rb') as f:
    run = cPickle.load(f)
frames = run['frames']
actions = run['actions']
action_meanings = run['action_meanings']

(h, w) = frames[0].shape[:2]

# create img for text
text_img = np.zeros((h/8, w, 3), dtype='uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
thikness = 1
font_size = 0.4

# Define the codec and create VideoWriter object
fourcc = cv2.cv.FOURCC(*args.codec)
out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h + h/8))


for i, (frame, action) in enumerate(zip(frames, actions)):
    # prepare text part
    text_img.fill(255)
    cv2.putText(text_img, 'action {}'.format(action_meanings[action]),
                (2, text_img.shape[0] / 2 - 2), font, font_size, (0, 0, 0), thikness)
    cv2.putText(text_img, 'frame {}'.format(i),
                (2, text_img.shape[0] - 2), font, font_size, (0, 0, 0), thikness)

    frame = np.vstack((frame, text_img))

    out.write(frame)

# Release everything if job is finished
out.release()
