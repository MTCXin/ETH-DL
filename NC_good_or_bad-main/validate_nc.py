import sys
import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import scipy.linalg as scilin

import models
from models.resnet import ResNet18
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10, MNIST
from data_loader.mini_imagenet import MiniImagenet
from utils import load_from_state_dict


class CIFAR10_subs(CIFAR10):
    def __init__(self, **kwargs):
        super(CIFAR10_subs, self).__init__(**kwargs)
        if self.train:
            with open(self.root+'/cifar10_uniform_128/train_label.pkl', 'rb') as f:
                train_all = pickle.load(f)
                self.targets = train_all["label"]
        else:
            with open(self.root+'/cifar10_uniform_128/test_label.pkl', 'rb') as f:
                test_all = pickle.load(f)
                self.targets = test_all["label"]

# def compute_accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
    
#     pred = output.view(batch_size, 128, 10)
#     pred = torch.linalg.norm(pred, dim=1)
    
#     _, pred = pred.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parse_eval_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')
    parser.add_argument('--do_adv', dest='do_adv', action='store_true')
    
    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=6)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)

    # Directory Setting
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar10_random', 'miniimagenet'], default='cifar10')
    parser.add_argument('--data_dir', type=str, default='/home/jinxin/data')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--p_name', type=str, default="info_new.pkl")

    # Learning Options
    parser.add_argument('--epochs', type=int, default=150, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size PER CLASS')

    args = parser.parse_args()

    return args

class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def split_array(input_array, batchsize=128):
    input_size = input_array.shape[0]
    num_splits, res_splits = input_size // batchsize, input_size % batchsize
    output_array_list = list()
    if res_splits == 0:
        output_array_list = np.split(input_array, batchsize, axis=0)
    else:
        for i in range(num_splits):
            output_array_list.append(input_array[i*batchsize:(i+1)*batchsize])

        output_array_list.append(input_array[num_splits*batchsize:])

    return output_array_list


def compute_info(args, model, fc_features, dataloader, do_adv = False):
    num_data = 0
    mu_G = 0
    mu_c_dict = dict()
    num_class_dict = dict()
    before_class_dict = dict()
    after_class_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Mean and Std transformation for doing adv training
    dmean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
    dstd = torch.tensor([0.2023, 0.1994, 0.2010]).to(device)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        
        if do_adv: # Need to do normalization if do_adv because this is not included in the dataloader
            inputs.sub_(dmean[None,:,None,None]).div_(dstd[None,:,None,None])

        with torch.no_grad():
            outputs = model(inputs)
            #fea, outputs = model(inputs)

        features = fc_features.outputs[0][0]
        # Need to normalize feature
        features = F.normalize(features, dim=1)
        # Need to normalize feature
        fc_features.clear()

        mu_G += torch.sum(features, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
                before_class_dict[y] = [features[b, :].detach().cpu().numpy()]
                after_class_dict[y] = [outputs[b, :].detach().cpu().numpy()]
                num_class_dict[y] = 1
            else:
                mu_c_dict[y] += features[b, :]
                before_class_dict[y].append(features[b, :].detach().cpu().numpy())
                after_class_dict[y].append(outputs[b, :].detach().cpu().numpy())
                num_class_dict[y] = num_class_dict[y] + 1

        num_data += targets.shape[0]

        prec1, prec5 = compute_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    mu_G /= num_data
    for i in range(len(mu_c_dict.keys())):
        mu_c_dict[i] /= num_class_dict[i]

    return mu_G, mu_c_dict, before_class_dict, after_class_dict, top1.avg, top5.avg


def compute_Sigma_W(args, before_class_dict, mu_c_dict, batchsize=128):
    num_data = 0
    Sigma_W = 0

    for target in before_class_dict.keys():
        class_feature_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        for features in class_feature_list:
            features = torch.from_numpy(features).to(args.device)
            Sigma_W_batch = (features - mu_c_dict[target].unsqueeze(0)).unsqueeze(2) * (
                        features - mu_c_dict[target].unsqueeze(0)).unsqueeze(1)
            Sigma_W += torch.sum(Sigma_W_batch, dim=0)
            num_data += features.shape[0]

    Sigma_W /= num_data
    return Sigma_W.detach().cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()

def compute_ETF(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


# def compute_nuclear_metric(all_features):
#     #all_features = info_pkl['before_class_dict_train'] # all features should be this
#     singular_values_dict = {} # each key contains the class's singular value array
#     for i in all_features: 
#         class_feature = np.array(all_features[i])
#         _,s,_ = np.linalg.svd(class_feature) # s is all singular values
#         singular_values_dict[i] = s
#     #print(len(singular_values_dict)) # sanity check
#     return singular_values_dict

def compute_nuclear_frobenius(all_features):
    #all_features = info_pkl['before_class_dict_train'] # all features should be this
    nf_metric_list = []
    for i in all_features: 
        class_feature = np.array(all_features[i])
        _,s,_ = np.linalg.svd(class_feature) # s is all singular values
        nuclear_norm = np.sum(s)
        frobenius_norm = np.linalg.norm(class_feature, ord='fro')
        nf_metric_class = nuclear_norm / frobenius_norm
        nf_metric_list.append(nf_metric_class)
    nf_metric = np.mean(nf_metric_list)
    return nf_metric


def compute_margin(args, before_class_dict, after_class_dict, W, b, mu_G, batchsize=128):
    num_data = 0
    avg_prob_margin = 0
    avg_cos_margin = 0
    all_prob_margin = list()
    all_cos_margin = list()

    for target in after_class_dict.keys():
        class_features_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        class_outputs_list = split_array(np.array(after_class_dict[target]), batchsize=batchsize)
        for i in range(len(class_outputs_list)):
            features, outputs = torch.from_numpy(class_features_list[i]).to(args.device), torch.from_numpy(class_outputs_list[i]).to(args.device)

            false_outputs = outputs.clone()
            false_outputs[:, target] = -np.inf
            false_targets = torch.argmax(false_outputs, dim=1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            prob_margin = probs[:, target] - torch.gather(probs, 1, false_targets.unsqueeze(1)).reshape(-1)
            all_prob_margin.append(prob_margin.detach().cpu().numpy())
            avg_prob_margin += torch.sum(prob_margin)

            cos_outputs = (outputs - b.unsqueeze(0)) / (torch.norm(features - mu_G.unsqueeze(0), dim=1, keepdim=True) * torch.norm(W.T, dim=0, keepdim=True))
            false_cos_outputs = cos_outputs.clone()
            false_cos_outputs[:, target] = -np.inf
            false_cos_targets = torch.argmax(false_cos_outputs, dim=1)

            cos_margin = cos_outputs[:, target] - torch.gather(false_cos_outputs, 1, false_cos_targets.unsqueeze(1)).reshape(-1)
            all_cos_margin.append(cos_margin.detach().cpu().numpy())
            avg_cos_margin += torch.sum(cos_margin)

            num_data += features.shape[0]

    avg_prob_margin /= num_data
    avg_cos_margin /= num_data
    all_prob_margin = np.sort(np.concatenate(all_prob_margin, axis=0))
    all_cos_margin = np.sort(np.concatenate(all_cos_margin, axis=0))

    prob_margin_dist_fig = plot_prob_margin_distribution(all_prob_margin)
    cos_margin_dist_fig = plot_cos_margin_distribution(all_cos_margin)
    return avg_prob_margin.item(), avg_cos_margin.item(), prob_margin_dist_fig, cos_margin_dist_fig


def compute_W_H_relation(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    M = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        M[:, i] = mu_c_dict[i] - mu_G
    sub = 1 / np.sqrt(K-1) * (torch.eye(K) - torch.ones(K, K) / K)
    
    ################
    #### Added  ####
#     if W.shape[0] == 1280:
#         W_clone = W.clone()
#         W_clone = W_clone.view(128,10,512)
#         W_clone = torch.mean(W_clone, 0)
#     WH = W_clone.cpu() @ M
    ################
    WH = W.cpu() @ M ####Uncomment this if delete above
    
    res = torch.norm(WH / torch.norm(WH, p='fro') - sub, p='fro')

    return res.detach().cpu().numpy()


def compute_Wh_b_relation(W, mu_G, b):
    Wh = torch.mv(W, mu_G.cuda())
    res_b = torch.norm(Wh + b, p='fro')
    return res_b.detach().cpu().numpy().item()


def plot_prob_margin_distribution(all_prob_margin):
    fig = plt.figure(figsize=(12, 8))
    plt.grid(True)

    plt.plot(all_prob_margin, 'c', linewidth=3, alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel('Probability Margin', fontsize=40)
    plt.title('Distibution of Probability Margin', fontsize=40)
    plt.xticks(np.arange(0,all_prob_margin.shape[0]+1,all_prob_margin.shape[0]//5), fontsize=30)

    plt.yticks(np.arange(-1, 1.01, 0.5), fontsize=30)

    plt.axis([0, all_prob_margin.shape[0], -1, 1])

    return fig

def plot_cos_margin_distribution(all_cos_margin):
    fig = plt.figure(figsize=(12, 8))
    plt.grid(True)

    plt.plot(all_cos_margin, 'c', linewidth=3, alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel('Cosine Margin', fontsize=40)
    plt.title('Distibution of Cosine Margin', fontsize=40)
    plt.xticks(np.arange(0, all_cos_margin.shape[0]+1, all_cos_margin.shape[0]//5), fontsize=30)

    plt.yticks(np.arange(-1.5, 1.51, 0.5), fontsize=30)

    plt.axis([0, all_cos_margin.shape[0], -1.5, 1.5])

    return fig


def main():
    args = parse_eval_args()

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    #device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    
    # Dataset part
    if args.dataset == "cifar10":
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
    elif args.dataset == "miniimagenet":
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(84, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        trainset = MiniImagenet(
            args.data_dir, "train", None, np.arange(50000), transform=transform_train, target_transform=None)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = MiniImagenet(
            args.data_dir, "test", None, None, transform=transform_test, target_transform=None)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
    else:
        print("That dataset is not yet implemented!")
    
    
    # Model part
    if args.ETF_fc:
        model = ResNet18(num_classes=num_classes,
                     norm_layer_type="bn",
                     conv_layer_type="conv",
                     linear_layer_type="linear",
                     activation_layer_type="relu",
                     etf_fc = True).to(device)
    else:
        model = ResNet18(num_classes=num_classes,
                     norm_layer_type="bn",
                     conv_layer_type="conv",
                     linear_layer_type="linear",
                     activation_layer_type="relu",
                     etf_fc = False).to(device)

    fc_features = FCFeatures()
    model.linear.register_forward_pre_hook(fc_features)

    info_dict = {
                 'collapse_metric': [],
                 'nuclear_metric': [],
                 'ETF_metric': [],
                 'WH_relation_metric': [],
                 'Wh_b_relation_metric': [],
                 'prob_margin': [],
                 'cos_margin': [],
                 'W': [],
                 'b': [],
                 'mu_G_train': [],
                 'mu_G_test': [],
                 'mu_c_dict_train': [],
                 'mu_c_dict_test': [],
                 'before_class_dict_train': {},
                 'after_class_dict_train': {},
                 'before_class_dict_test': {},
                 'after_class_dict_test': {},
                 'train_acc1': [],
                 'train_acc5': [],
                 'test_acc1': [],
                 'test_acc5': []
                 }

    for i in range(args.epochs):
#         if i == 0:
#             state_d = torch.load(args.load_path + 'model_epoch_' + str(i + 1) + '.pth', map_location=device)["state_dict"]
#             load_from_state_dict(model, state_d)
#         else:
#             model.load_state_dict(torch.load(args.load_path + 'model_epoch_' + str(i + 1) + '.pth', map_location=device)["state_dict"])
        model.load_state_dict(torch.load(args.load_path + 'model_epoch_' + str(i + 1) + '.pth', map_location=device)["state_dict"])
    #### Indentation
        #model.load_state_dict(torch.load(args.load_path, map_location=device)["state_dict"])
        model.eval()

        for n, p in model.named_parameters():
            if 'linear.weight' in n:
                W = p.clone()
            if 'linear.bias' in n:
                b = p.clone()

        if not args.bias:
            b = torch.zeros((W.shape[0],), device=device)

        mu_G_train, mu_c_dict_train, before_class_dict_train, after_class_dict_train, train_acc1, train_acc5 = compute_info(args, model, fc_features, trainloader, do_adv=args.do_adv)
        mu_G_test, mu_c_dict_test, before_class_dict_test, after_class_dict_test, test_acc1, test_acc5 = compute_info(args, model, fc_features, testloader, do_adv=args.do_adv)

        Sigma_W = compute_Sigma_W(args, before_class_dict_train, mu_c_dict_train, batchsize=args.batch_size)
        # Sigma_W_test_norm = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, testloader, isTrain=False)
        Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)

        collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
        ETF_metric = compute_ETF(W)
        
        # Add for nuclear metric
        #nuclear_epoch = compute_nuclear_metric(before_class_dict_train)
        nf_metric_epoch = compute_nuclear_frobenius(before_class_dict_train)
        info_dict['nuclear_metric'].append(nf_metric_epoch)

        avg_prob_margin, avg_cos_margin, prob_margin_dist_fig, cos_margin_dist_fig = \
            compute_margin(args, before_class_dict_train, after_class_dict_train, W, b, mu_G_train, batchsize=args.batch_size)
        info_dict['prob_margin'].append(avg_prob_margin)
        info_dict['cos_margin'].append(avg_cos_margin)
    
        WH_relation_metric = compute_W_H_relation(W, mu_c_dict_train, mu_G_train) # Added back
        Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, b)

        info_dict['collapse_metric'].append(collapse_metric)
        info_dict['ETF_metric'].append(ETF_metric)
        info_dict['WH_relation_metric'].append(WH_relation_metric) # Added back
        info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)
        info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())
        info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())
        info_dict['mu_c_dict_train'] = mu_c_dict_train
        info_dict['mu_c_dict_test'] = mu_c_dict_test
        info_dict['before_class_dict_train'] = before_class_dict_train
        info_dict['after_class_dict_train'] = after_class_dict_train
        info_dict['before_class_dict_test'] = before_class_dict_test
        info_dict['after_class_dict_test'] = after_class_dict_test
        info_dict['W'].append((W.detach().cpu().numpy()))
        if args.bias:
            info_dict['b'].append(b.detach().cpu().numpy())

        info_dict['train_acc1'].append(train_acc1)
        info_dict['train_acc5'].append(train_acc5)
        info_dict['test_acc1'].append(test_acc1)
        info_dict['test_acc5'].append(test_acc5)


        if not os.path.exists(args.load_path + 'probability_margin/'):
            os.mkdir(args.load_path + 'probability_margin/')
        prob_margin_dist_fig.savefig(args.load_path + 'probability_margin/' + "p_margin_epochs_%3d.pdf" %i,
                                     bbox_inches='tight')


        if not os.path.exists(args.load_path + 'cosine_margin/'):
            os.mkdir(args.load_path + 'cosine_margin/')
        prob_margin_dist_fig.savefig(args.load_path + 'cosine_margin/' + "c_margin_epochs_%3d.pdf" %i,
                                     bbox_inches='tight')
        print(f"Epoch {i} is processed")
        
    with open(args.load_path + args.p_name, 'wb') as f: #'info_normal.pkl'
        pickle.dump(info_dict, f)



if __name__ == "__main__":
    main()
