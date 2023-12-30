# CREATED BY RUIZHE ZHU, 10/12/2023

import os
import json
from tqdm import tqdm
from metrics import *
import pdb
# METRICS = [Color_Histogram, Color_Moment, Dominant_Color_Descriptor, Gray_Level_Cooccurrence_Matrix, 
#            Local_Binary_Patterns, Gabor_Filters, Histogram_of_Oriented_Gradients, Sobel, Prewitt, 
#            Canny, Laplacian_of_Gaussian_Filter, Entropy, Fractal_Dimension, Edge_Density, 
#            Spatial_Information, Discrete_Fourier_Transform, Wavelet_Transform, Histogram_Equalization,
#            Discrete_Cosine_Transform, SVD, PCA_transform]
METRICS = [Histogram_Equalization, Discrete_Cosine_Transform_max, Discrete_Cosine_Transform_avg, SVD, PCA_transform]
PATH = './imgs/'
JSON_NAME = 'TEST'
GT_JSON='./result_black_simba.json'

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
    with open(GT_JSON,'r') as load_f:
        res_dict = json.load(load_f)

    for file in tqdm(res_dict.keys()):
        parts = file.split('/')
        img_dataset, img_class = parts[2],parts[3]
        json_add(img_dataset, img_class, file, json_dic)
    with open(JSON_NAME+'.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_dic, json_file)
   
if __name__ == '__main__':
    main()
