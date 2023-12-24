# ETH-DL
Deep Learning project


## TODO: 
- XIN: 继续跑出更多结果
## Xin-black-attack-simba
- 黑盒攻击
- 目前使用imagenet1k - valid部分https://www.kaggle.com/datasets/sautkin/imagenet1kvalid/
- 目前结果基于Resnet18模型，更改模型/数据集 简单
- 结果存于文件夹下.\result_black_simba.json 其中linf_norm无效（由于攻击算法linf总是一样的），queries代表攻破图片时向模型请求了多少次output。probs代表攻破时正确类别置信度
- 目前结果还在继续跑，大约每小时200个。
## get_metrics.py
- 特征提取框架

## metrics.py
- 各种特征提取算法

## test.ipynb
- img_feature.json一个特征存一个文件
