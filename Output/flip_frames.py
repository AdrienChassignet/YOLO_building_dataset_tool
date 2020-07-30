import os
import glob
import cv2
import numpy as np

run_dir = os.path.dirname(os.path.abspath(__file__))
print(run_dir)

for f in glob.glob(run_dir+'/**/*.jpg', recursive=True):
    image = cv2.imread(f)
    image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(f, image)

    with open(os.path.splitext(f)[0]+'.txt', 'r') as labelfile:
        lines = []
        for (i, line) in enumerate(labelfile):
            (id, x, y, w, h) = line.split()
            x = 1-float(x)
            y = 1-float(y)
            lines.append('{0} {1} {2} {3} {4}'.format(id, x, y, w, h))
    with open(os.path.splitext(f)[0]+'.txt', 'w') as labelfile:
        labelfile.writelines(line for line in lines)
