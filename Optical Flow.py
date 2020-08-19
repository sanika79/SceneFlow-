#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import flow_vis
from tqdm.notebook import tqdm

def load3x3(fp):
    mat = []
    with open(fp) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            mat.extend(line.split(','))
    return np.array(mat).reshape(3, 3).astype(np.float)

R = load3x3("data/pair_3_rot.txt")
left_dir = "data/18095501_3L/"
right_dir = "data/180E5501_3R/"
frames = []
for num in tqdm(range(480, 999)):
    file_t = left_dir + f"3L_0{num}.jpg"
    file_t_1 = left_dir + f"3L_0{num + 1}.jpg"
    t_0 = cv2.imread(file_t, 0)
    t_1 = cv2.imread(file_t_1, 0)
    flow = cv2.calcOpticalFlowFarneback(t, t_1, None, 0.5, 3, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    frames.append(flow_color)
size = t_0.shape
video = cv2.VideoWriter("project.avi",cv2.VideoWriter_fourcc(*"MPEG"), 24, size[::-1])
print(video.isOpened())
for frame in tqdm(frames):
    video.write(frame)
video.release()


# In[ ]:




