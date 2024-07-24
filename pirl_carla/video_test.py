#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:19:15 2024

@author: ubuntu-root
"""

import sys
import glob
import cv2

# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output file name, encoder, fps, size(fit to image size)
video = cv2.VideoWriter('video.mp4',fourcc, 20.0, (640,480))


if not video.isOpened():
    print("can't be opened")
    sys.exit()

    
for filename in sorted(glob.glob("video/*.png")):    

    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv2.imread(filename)

    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)
    print(filename)

video.release()
print('written')