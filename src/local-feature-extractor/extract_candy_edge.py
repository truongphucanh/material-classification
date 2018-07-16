# python ./material_classification/source/local-feature-extractor/extract_candy_edge.py <dataset_name> 
import cv2
import numpy as np
import os
import sys

dataset_name = sys.argv[1
data_folder = "./material_classification/source/DS_{}/data/original".format(dataset_name)
for root, dirs, files in os.walk(data_folder):
    for filename in [f for f in files if f.endswith(".jpg")]:
		image = cv2.imread(filename, 0)
		edges = cv2.Canny(image,100,200)
		out_file = filename.replace("original", "edges")
		cv2.imwrite(out_file, edges)
