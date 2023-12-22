# CREATED BY RUIZHE ZHU, 10/12/2023

import os
import json
from metrics import *

METRICS = [Color_Histogram, Color_Moment, Dominant_Color_Descriptor, Gray_Level_Cooccurrence_Matrix, 
           Local_Binary_Patterns, Gabor_Filters, Histogram_of_Oriented_Gradients, Sobel, Prewitt, 
           Canny, Laplacian_of_Gaussian_Filter, Entropy, Fractal_Dimension, Edge_Density, 
           Spatial_Information, Discrete_Fourier_Transform, Wavelet_Transform, Histogram_Equalization,
           Discrete_Cosine_Transform, SVD, PCA_transform]

PATH = './imgs/'
JSON_NAME = 'TEST'

def feature_extract(img_path):
    metric_dic = {}
    for metric in METRICS:
        metric_dic[metric.__name__] = metric(img_path).tolist()
    return metric_dic

def json_add(img_dataset, img_class, img_path, json_dic):
    image_dic = {}
    image_dic['DATASET'] = img_dataset
    image_dic['CLASS'] = img_class
    image_dic['METRICS'] = feature_extract(img_path)
    json_dic[img_path] = image_dic

def main():
    json_dic = {}
    for root, ds, fs in os.walk(PATH):
        for file in fs:
            img_dataset, img_class = root.replace(PATH, '').split('\\')
            img_path = os.path.join(root, file)
            json_add(img_dataset, img_class, img_path, json_dic)
    with open(JSON_NAME+'.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_dic, json_file, indent=4)
   
if __name__ == '__main__':
    main()
