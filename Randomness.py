import os
import json
import numpy as np

Attack = {}

PATH = './'
for root, ds, fs in os.walk(PATH):
    if 'diff' in root and '__' not in root and '0' in root:
        jfile = fs[-1]
        print(fs)
        f = open(os.path.join(root,jfile), encoding='utf-8') 
        result = json.load(f)
        f.close()
        for key in result.keys():
            if key not in Attack.keys():
                Attack[key] = {}
                Attack[key]['l2_norm'] = [] 
                Attack[key]['queries'] = [] 
            if result[key]['l2_norm'] == 0:
                result[key]['l2_norm'] = 15
                result[key]['queries:'] = 24000
            Attack[key]['l2_norm'].append(result[key]['l2_norm'])
            Attack[key]['queries'].append(result[key]['queries:'])
      
var = []       
a = 0

varq = []

for key in Attack.keys():
    l2norm = Attack[key]['l2_norm']
    # print(l2norm)
    # a += np.count_nonzero(l2norm)
    if np.std(l2norm) != 0:
        var.append(np.std(l2norm))
        varq.append(np.std(Attack[key]['queries']))

print(len(var))
print(np.mean(var), np.mean(varq))
print(np.std(var), np.std(varq))
print(np.count_nonzero(var))
print(a)