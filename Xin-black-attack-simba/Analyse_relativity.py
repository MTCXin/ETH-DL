import matplotlib.pyplot as plt
import numpy as np
import json
import pdb
with open('result_black_simba.json','r') as load_f:
    res_dict_resnet18 = json.load(load_f)
with open('result_black_simba——effnet.json','r') as load_f:
    res_dict_effnet = json.load(load_f)

toplotl2norm_x=[]
toplotquery_x=[]
toplotl2norm_y=[]
toplotquery_y=[]
toplotprob=[]
for key,value in res_dict_effnet.items():
    if key in res_dict_resnet18.keys() and value["l2_norm"]!=0 and value["queries:"]!=0 and res_dict_resnet18[key]["queries:"]!=0 and res_dict_resnet18[key]["l2_norm"]!=0:
        toplotl2norm_x.append(value["l2_norm"])
        toplotquery_x.append(value["queries:"])
        toplotl2norm_y.append(res_dict_resnet18[key]["l2_norm"])
        toplotquery_y.append(res_dict_resnet18[key]["queries:"])
        # toplotprob.append([value["probs"],res_dict_resnet18[key]["probs"]])
# fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)
# # pdb.set_trace()
# axs[0].scatter(toplotl2norm_x,toplotl2norm_y, s=1)
# axs[0].set_title("l2")
# axs[1].scatter(toplotquery_x,toplotquery_y, s=1)
# axs[1].set_title("query")

# plt.show()
corr_coefficient_l2 = np.corrcoef(toplotl2norm_x, toplotl2norm_y)[0, 1]
corr_coefficient_query = np.corrcoef(toplotquery_x, toplotquery_y)[0, 1]

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for l2 norms
axs[0].scatter(toplotl2norm_x, toplotl2norm_y, s=1)
axs[0].set_title(f"L2 Norm Correlation (r = {corr_coefficient_l2:.2f})")
axs[0].set_xlabel('X axis - Effnet l2_norm')
axs[0].set_ylabel('Y axis - Resnet18 l2_norm')

# Plot for queries
axs[1].scatter(toplotquery_x, toplotquery_y, s=1)
axs[1].set_title(f"Query Correlation (r = {corr_coefficient_query:.2f})")
axs[1].set_xlabel('X axis - Effnet queries')
axs[1].set_ylabel('Y axis - Resnet18 queries')

# Display the plot
plt.show()