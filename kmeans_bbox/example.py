import os

import numpy as np

from kmeans_bbox.kmeans import kmeans, avg_iou

def yolo2voc(box,img_w,img_h):
	center_x = round(float(box[0]) * img_w)
	center_y = round(float(box[1]) * img_h)
	bbox_width = round(float(box[2]) * img_w)
	bbox_height = round(float(box[3]) * img_h)  # 框 的 中心坐标 + 宽高 yolo

	xmin = int(center_x - bbox_width / 2)
	ymin = int(center_y - bbox_height / 2)
	xmax = int(center_x + bbox_width / 2)
	ymax = int(center_y + bbox_height / 2)
	return xmin,ymin,xmax,ymax


all_txt = "F:\Projects\yolov3\sl-master\data131_v1\\target\\"
paths = os.listdir(all_txt)
img_w = img_h = 512
data = []
CLUSTERS = 12
for path in paths:
	bbox_path = all_txt + path
	bbox_loc = np.loadtxt(bbox_path)
	if bbox_loc.shape[0]==2:
		loc_one = bbox_loc[0,1:5]
		loc_two = bbox_loc[1,1:5]
		xmin_one,ymin_one,xmax_one,ymax_one = yolo2voc(loc_one,img_w,img_h)
		xmin_two, ymin_two, xmax_two, ymax_two = yolo2voc(loc_one, img_w, img_h)
		data.append([xmax_one - xmin_one, ymax_one - ymin_one])
		data.append([xmax_two - xmin_two, ymax_two - ymin_two])
	elif bbox_loc.shape[0]==5 and bbox_loc[1]!=0 and bbox_loc[2]!=0:
		loc_one = bbox_loc[1:5]
		xmin_one, ymin_one, xmax_one, ymax_one = yolo2voc(loc_one, img_w, img_h)
		data.append([xmax_one - xmin_one, ymax_one - ymin_one]) # width,height
	else:
		print(bbox_path,bbox_loc[1],bbox_loc[2])



data = np.array(data)
print(data)

out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))


