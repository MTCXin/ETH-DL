# ETH-DL
Deep Learning project

## Xin-black-attack-simba
- Main code of black-box attack
## simba_results / result_black_simbav2.json
- Results of black-box attacks
## Randomness.py
- Results analysis
## metrics.py
- Algorithms of image features
## get_metrics.py
- Get image features
## Regression.Rmd
- Regression Analysis
## feature_analysis.ipynb
- Feature Analysis
## Vulnet.py/Vulnet_classify.py/MetricsNet.py
- Neural Networks

# How to run the code
Our dataset is a part of widely used ImageNet1K dataset and it is available from. Xin-black-simba-attack folder contains code about the Simba algorithm. Applying the Simba algorithm and then we will get results shown in simba_results. On the other hand, image features shown in metrics.py can be exctracted by get_metrics.py. Preprocessing.py helps merge the two json files and get X.csv and Y.csv which are used for analysis. Regression.rmd is a R markdown file for Regression Analysis while feature_analysis.ipynb shows some other analysis. Vulnet.py, Vulnet_classify.py and MetricsNet.py are some Neural Network models we built.
