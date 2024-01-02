# CREATED BY RUIZHE ZHU, 10/12/2023

import os
import json
from tqdm import tqdm
from metrics import *

f = open('result_black_simba.json', encoding='utf-8') 
attack_result = json.load(f)
f.close()
avail_pics = attack_result.keys()


METRICS = [Color_Statistic, Dominant_Color_Descriptor, Gray_Level_Cooccurrence_Matrix,
           Local_Binary_Patterns, Gabor_Filters, Sobel, Canny,
           Entropy, Fractal_Dimension,
           SVD, PCA_transform]

# METRICS = [Local_Binary_Patterns]

PATH = './imgs/'
JSON_NAME = 'TEST'

def feature_extract(img_path):
    metric_dic = {}
    for metric in METRICS:
        metricres = metric(img_path)
        metric_dic[metric.__name__] = metricres.tolist()
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
        for file in tqdm(fs):
            img_dataset, img_class = root.replace(PATH, '').split('\\')
            img_path = os.path.join(root, file)
            if img_path.replace('\\', '/') in avail_pics:
                json_add(img_dataset, img_class, img_path, json_dic)
    with open(JSON_NAME+'.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_dic, json_file, indent=4)
   
   
if __name__ == '__main__':
    main()