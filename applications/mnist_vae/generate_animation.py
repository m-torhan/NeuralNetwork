import matplotlib.pyplot as plt
import cv2
import numpy as np

gen_along_line_data = []
points_per_label = {i : [[], []] for i in range(10)}

with open('./data/test_encoded.txt', 'r') as input_file:
    for line in input_file:
        label, x, y = line.split(';')
        label = int(label)
        x = float(x)
        y = float(y)
        
        points_per_label[label][0].append(x)
        points_per_label[label][1].append(y)

with open('./data/gen_along_line.txt', 'r') as input_file:
    for line in input_file:
        x, y, *img = list(map(float, line.split(';')))

        img = np.array(img).reshape(28, 28)

        gen_along_line_data.append(((x, y), img))


plt.figure(figsize=(16, 16))

for l in range(label):
    plt.scatter(points_per_label[l][0], points_per_label[l][1])

plt.ylim(-4, 4)
plt.xlim(-4, 4)

plt.savefig('test.png')
