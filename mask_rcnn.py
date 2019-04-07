import os
import cv2
import utils
import random
import argparse
import numpy as np

def get_args(video=False):
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image_path", required=not video,
		help="path to input image")
	ap.add_argument("-m", "--video_path", required=video,
		help="path to input video")
	ap.add_argument("-p", "--classes_path", required=True,
		help="path to object_detection_class_person.txt")
	ap.add_argument("-e", "--inception", required=True,
		help="path to mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
	ap.add_argument("-f", "--frozen_inference", required=True,
		help="path to frozen_inference_graph.pb")
	ap.add_argument("-o", "--output", required=video,
		help="path to frozen_inference_graph.pb")
	ap.add_argument("-c", "--confidence", type=float, default=0.65,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
		help="minimum threshold for pixel-wise mask segmentation")
		
	return vars(ap.parse_args())

COLORS = utils.COLORS

def get_labels(path):
	return open(path).read().strip().split("\n")

def get_net(weights_path, config_path):
	return cv2.dnn.readNetFromTensorflow(weights_path, config_path)

def process_image(nn, image):
	blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
	nn.setInput(blob)
	return nn.forward(["detection_out_final", "detection_masks"])

def blur_background(image, boxes, masks, labels, confidence=0.5, threshold=0.3):
	clone = image.copy()
	blurred_clone = cv2.GaussianBlur(image, (51, 51), 0)
	rois = []
	for i in range(0, boxes.shape[2]):
		classID = int(boxes[0, 0, i, 1])
		score = boxes[0, 0, i, 2]
		if score > confidence:
			(H, W) = clone.shape[:2]
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])

			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_CUBIC)
			mask = (mask > threshold)
			
			clone_bg = np.ones_like(clone)
			clone_bg[startY:endY, startX:endX][mask] = np.array([255,255,255])
			blurred_clone[startY:endY, startX:endX][mask] = clone[startY:endY, startX:endX][mask] 
	return blurred_clone