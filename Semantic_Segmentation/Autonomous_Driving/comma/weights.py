import os
import cv2
import numpy as np
from math import ceil
from utils import rgb2cat, label_defs

data_dir = 'data'
batch = 10

with open(os.path.join(data_dir, 'train/files'), 'r') as file:
    lines = [os.path.join(data_dir, 'train/masks', line.strip()) for line in file.readlines()]

count = np.arange(ceil(len(lines)/batch))
avg = []

for i in count:

    temp = np.arange(start=i*10, stop=min((i+1)*10, len(lines)), step=1)
    mean = []
    images = []

    for index in temp:
        images.append(cv2.cvtColor(cv2.imread(lines[index]), cv2.COLOR_BGR2RGB))

    images = rgb2cat(images)
    images = np.argmax(images, axis = -1) 
    labels, counts = np.unique(images, return_counts=True)

    for freq in counts:
        mean.append(freq/images.size)

    avg.append(mean)

avg = np.mean(np.array(avg), axis=0)

with open('weights', 'w') as file:
    for i, label in enumerate(label_defs):
        file.write(label[0] + ' : ' + str(avg[i]) + '\n')