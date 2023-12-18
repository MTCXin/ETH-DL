# CREATED BY RUIZHE ZHU, 10/12/2023
import os
import json
from metrics import *
import warnings
warnings.filterwarnings("ignore")

METRICS = [Color_Moment,
            Gray_Level_Cooccurrence_Matrix, 
            Entropy, Fractal_Dimension,
            Discrete_Cosine_Transform, SVD]
PATH = './imgs/'
JSON_NAME = 'image_features'

def feature_extract(img_path):
    metric_dic = {}
    for metric in METRICS:
        metric_dic[metric.__name__] = metric(img_path).tolist()
    return metric_dic

def json_add(img_dataset, img_class, img_path, metric_dict, json_array):
    json_dic = {}
    json_dic['DATASET'] = img_dataset
    json_dic['PATH'] = img_path
    json_dic['CLASS'] = img_class
    json_dic['METRICS'] = metric_dict
    json_array.append(json_dic)

def main():
    for root, ds, fs in os.walk(PATH):
        json_array = []
        for file in fs:
            img_dataset, img_class = root.replace(PATH, '').split('\\')
            img_path = os.path.join(root, file)
            img_path = img_path.replace('\\', '/')
            print(img_path)
            metric_dic = feature_extract(img_path)
            json_add(img_dataset, img_class, img_path, metric_dic, json_array)
        with open(JSON_NAME+'.json', 'a', encoding='utf-8') as json_file:
            json.dump(json_array, json_file, indent=4)
    
if __name__ == '__main__':
    main()