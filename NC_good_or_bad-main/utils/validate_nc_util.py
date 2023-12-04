import sys
import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import scipy.linalg as scilin

import copy
import os
import numpy as np
from utils import load_from_state_dict
import matplotlib.pyplot as plt

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


def compute_info(device, model, fc_features, dataloader, do_adv=False):

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

        inputs, targets = inputs.to(device), targets.to(device)
        
        if do_adv: # Need to do normalization if do_adv because this is not included in the dataloader
            inputs.sub_(dmean[None,:,None,None]).div_(dstd[None,:,None,None])

        with torch.no_grad():
            outputs = model(inputs)

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


def compute_Sigma_W(device, before_class_dict, mu_c_dict, batchsize=128):
    
    num_data = 0
    Sigma_W = 0

    for target in before_class_dict.keys():
        class_feature_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        for features in class_feature_list:
            features = torch.from_numpy(features).to(device)
            Sigma_W_batch = (features - mu_c_dict[target].unsqueeze(0)).unsqueeze(2) * (features - mu_c_dict[target].unsqueeze(0)).unsqueeze(1)
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

    return Sigma_B.detach().cpu().numpy()


def compute_ETF(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_Wh_b_relation(W, mu_G, b):
    Wh = torch.mv(W, mu_G.cuda())
    res_b = torch.norm(Wh + b, p='fro')
    return res_b.detach().cpu().numpy().item()

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

def compute_margin(device, before_class_dict, after_class_dict, W, b, mu_G, batchsize=128):
    num_data = 0
    avg_prob_margin = 0
    avg_cos_margin = 0
    all_prob_margin = list()
    all_cos_margin = list()

    for target in after_class_dict.keys():
        class_features_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        class_outputs_list = split_array(np.array(after_class_dict[target]), batchsize=batchsize)
        for i in range(len(class_outputs_list)):
            features, outputs = torch.from_numpy(class_features_list[i]).to(device), torch.from_numpy(class_outputs_list[i]).to(device)

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

    WH = W.cpu() @ M ####Uncomment this if delete above
    
    res = torch.norm(WH / torch.norm(WH, p='fro') - sub, p='fro')

    return res.detach().cpu().numpy()


def validate_nc_epoch(checkpoint_dir, epoch, orig_model, trainloader, testloader, info_dict, do_adv = False):
    print(f"Processing the NC information for epoch {epoch}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(orig_model)
    batchsize = 1024
    
    fc_features = FCFeatures()
    model.linear.register_forward_pre_hook(fc_features)

#     model.load_state_dict(torch.load(str(checkpoint_dir / f'model_epoch_{epoch}.pth'), map_location=device)["state_dict"])
    model.eval()
    
    have_bias = False
    for n, p in model.named_parameters():
        if 'linear.weight' in n:
            W = p.clone()
        if 'linear.bias' in n:
            b = p.clone()
            have_bias = True

    if not have_bias:
        b = torch.zeros((W.shape[0],), device=device)

    mu_G_train, mu_c_dict_train, before_class_dict_train, after_class_dict_train, train_acc1, train_acc5 = compute_info(device, model, fc_features, trainloader, do_adv = do_adv)
    mu_G_test, mu_c_dict_test, before_class_dict_test, after_class_dict_test, test_acc1, test_acc5 = compute_info(device, model, fc_features, testloader, do_adv = do_adv)
    
    Sigma_W = compute_Sigma_W(device, before_class_dict_train, mu_c_dict_train, batchsize=batchsize)
    
    Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)

    collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
    ETF_metric = compute_ETF(W)
    # Add for nuclear metric
    #nuclear_epoch = compute_nuclear_metric(before_class_dict_train)
    nf_metric_epoch = compute_nuclear_frobenius(before_class_dict_train)
    info_dict['nuclear_metric'].append(nf_metric_epoch)

    # Add for prob margin and cos margin
    avg_prob_margin, avg_cos_margin, prob_margin_dist_fig, cos_margin_dist_fig = \
        compute_margin(device, before_class_dict_train, after_class_dict_train, W, b, mu_G_train, batchsize=batchsize)
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
    for key in mu_c_dict_train:
        mu_c_dict_train[key] = mu_c_dict_train[key].detach().cpu().numpy()
    for key in mu_c_dict_test:
        mu_c_dict_test[key] = mu_c_dict_test[key].detach().cpu().numpy()
    info_dict['mu_c_dict_train'] = mu_c_dict_train
    info_dict['mu_c_dict_test'] = mu_c_dict_test
    info_dict['before_class_dict_train'] = before_class_dict_train
    info_dict['after_class_dict_train'] = after_class_dict_train
    info_dict['before_class_dict_test'] = before_class_dict_test
    info_dict['after_class_dict_test'] = after_class_dict_test
    info_dict['W'].append((W.detach().cpu().numpy()))
    if have_bias:
        info_dict['b'].append(b.detach().cpu().numpy())

    info_dict['train_acc1'].append(train_acc1)
    info_dict['train_acc5'].append(train_acc5)
    info_dict['test_acc1'].append(test_acc1)
    info_dict['test_acc5'].append(test_acc5)

    print(f"Epoch {epoch} is processed")
    
    return collapse_metric, nf_metric_epoch, ETF_metric, WH_relation_metric, Wh_b_relation_metric,\
           avg_prob_margin, avg_cos_margin, prob_margin_dist_fig, cos_margin_dist_fig
        

def plot_nc(info_dict, epochs):
    XTICKS = [30 * i for i in range(8) if i < epochs / 30]
    
    fig_collapse = plot_collapse(info_dict, epochs, XTICKS)
    fig_nuclear = plot_nuclear(info_dict, epochs, XTICKS)
    fig_etf = plot_ETF(info_dict, epochs, XTICKS)
    fig_wh = plot_WH_relation(info_dict, epochs, XTICKS)
    fig_whb = plot_WH_b_relation(info_dict, epochs, XTICKS)
    fig_prob_margin = plot_prob_margin(info_dict, epochs, XTICKS)
    fig_cos_margin = plot_cos_margin(info_dict, epochs, XTICKS)
    fig_train_acc = plot_train_acc(info_dict, epochs, XTICKS)
    fig_test_acc = plot_test_acc(info_dict, epochs, XTICKS)
    
    return fig_collapse, fig_nuclear, fig_etf, fig_wh, fig_whb,\
           fig_prob_margin, fig_cos_margin, fig_train_acc, fig_test_acc
    

############ Below are support methods for plot_nc ############
###############################################################
def plot_collapse(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    
    plt.plot(info['collapse_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 2, 0.1), fontsize=30) 

    plt.axis([0, epochs, 0, 1.0]) 
    
    return fig

def plot_ETF(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['ETF_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(-0.2, 1.21, .2), fontsize=30) 
    
    plt.axis([0, epochs, -0.02, 1.2]) 

    return fig

# def plot_nuclear(info, epochs, XTICKS):
#     fig = plt.figure(figsize=(10, 8))
#     plt.grid(True)
#     all_singular = info['nuclear_metric']
#     nuclear_norm_list = [] # final outcome, # of epochs list contain average nuclear norm of all classes, each epoch
    
#     for i in range(len(all_singular)):
#         epoch_nuclear = all_singular[i]
#         temp = [] #store sum of singular values of different class
#         for j in epoch_nuclear: # for each class
#             s = epoch_nuclear[j]
#             s /= np.max(s) # normalize by spectral norm
#             temp.append(np.sum(s))
#         nuclear_norm_list.append(np.mean(temp)) # mean value of nuclear norm of each class
#     print(len(nuclear_norm_list))
#     plt.plot(nuclear_norm_list, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

#     plt.xlabel('Epoch', fontsize=40)
#     plt.ylabel('Avg. Nuclear Norm (feature)', fontsize=40)
#     plt.xticks(XTICKS, fontsize=30)

#     ymax = np.max(nuclear_norm_list) + 5
#     plt.yticks(np.arange(0, ymax, int(ymax/5)), fontsize=30) 
    
#     plt.axis([0, epochs, 0, ymax])  

#     return fig

def plot_nuclear(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    all_nf_metric = info['nuclear_metric']
    
    plt.plot(all_nf_metric, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Avg. class NF_metric', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    ymax = np.max(all_nf_metric) + 0.5
    plt.yticks(np.arange(0, ymax, int(ymax/5)), fontsize=30) 
    
    plt.axis([0, epochs, 0, ymax])  

    return fig

def plot_WH_relation(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['WH_relation_metric'], 'c', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 1.21, 0.2), fontsize=30)

    plt.axis([0, epochs, 0, 1.2]) 

    return fig

def plot_WH_b_relation(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['Wh_b_relation_metric'], 'c', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 8.01, 2), fontsize=30)

    plt.axis([0, epochs, 0, 8])

    return fig

def plot_prob_margin(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['prob_margin'], 'c', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{P}_{M}$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(-0.2, 1.01, 0.2), fontsize=30)

    plt.axis([0, epochs, -0.2, 1])

    return fig

def plot_cos_margin(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['cos_margin'], 'c', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{cos}_{M}$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(-0.3, 1.21, 0.25), fontsize=30)

    plt.axis([0, epochs, -0.3, 1.2])

    return fig

def plot_train_acc(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['train_acc1'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(20, 110, 20), fontsize=30) 

    plt.axis([0, epochs, 20, 102])

    return fig

def plot_test_acc(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    
    plt.plot(info['test_acc1'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(20, 100.1, 10), fontsize=30)

    plt.axis([0, epochs, 20, 100])

    return fig

def plot_prob_margin_distribution(all_prob_margin):
    fig = plt.figure(figsize=(10, 8))
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
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(all_cos_margin, 'c', linewidth=3, alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel('Cosine Margin', fontsize=40)
    plt.title('Distibution of Cosine Margin', fontsize=40)
    plt.xticks(np.arange(0, all_cos_margin.shape[0]+1, all_cos_margin.shape[0]//5), fontsize=30)

    plt.yticks(np.arange(-1.5, 1.51, 0.5), fontsize=30)

    plt.axis([0, all_cos_margin.shape[0], -1.5, 1.5])

    return fig
