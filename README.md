# ETH-DL
Deep Learning project
先翻出来去年用过的代码，接下来把它们改成直接在cifar数据集上运行的代码

## TODO: 
- 对Cifar提取特征并保存为JSON
- 对Cifar进行攻击并保存范数信息
## simple-blackbox
- 黑盒攻击
- 运行run_simba.py / run_simba_cifar.py
- 目前使用imagenet1k - valid部分https://www.kaggle.com/datasets/sautkin/imagenet1kvalid/

## NC_Good_or_Bad
- 白盒攻击
- 使用validate_pgd.py / fgsm.py
