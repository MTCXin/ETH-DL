import pandas as pd
import numpy as np
import json

f1 = open('result_black_simba.json', encoding='utf-8') 
attack_result = json.load(f1)
f1.close()
f2 = open('Metrics.json', encoding='utf-8') 
metrics = json.load(f2)
f2.close()

length_dic = {}
for key in metrics.keys():
    for key2 in metrics[key]['METRICS'].keys():
        currentmetric = np.array(metrics[key]['METRICS'][key2]).reshape(-1)
        length_dic[key2] = currentmetric.shape[0]
    break

names = []
for key in length_dic.keys():
    if length_dic[key] < 300:
        for i in range(1, length_dic[key] + 1):
            names.append(f"{key}_{i}")

X = pd.DataFrame(columns=names)
Y = pd.DataFrame(columns=['l2_norm', 'linf_norm'])           

for key in attack_result.keys():

    Y.loc[len(Y)] = [attack_result[key]['l2_norm'], attack_result[key]['linf_norm']]
    
    values = np.array([])
    for key2 in metrics[key]['METRICS'].keys():
        currentvalue = np.array(metrics[key]['METRICS'][key2]).reshape(-1)
        if key2 != 'SVD':
            values = np.concatenate([values, currentvalue])
    #     print(key2, currentvalue.shape[0])
    # print(values.shape) 
    X.loc[len(X)] = values
    
X.to_csv('X.csv')
Y.to_csv('Y.csv')