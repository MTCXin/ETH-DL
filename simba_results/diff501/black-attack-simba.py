import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torchvision.models as models
from torchvision.models import resnet18,ResNet18_Weights
import simba_utils
import math
import random
import argparse
import os
from simba import SimBA
import glob  
import json
import pdb

random.seed(42)
torch.manual_seed(40)
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 

# parser nohup python get_metrics_xin_copy.py > output-getmetric.log &

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--model', type=str, default='resnet50', help='https://pytorch.org/vision/stable/models.html')
parser.add_argument('--data_root', type=str, default='./../imgs/ImageNet1000', help='root directory of imagenet data')
parser.add_argument('--num_runs', type=int, default=500, help='number of image samples')
parser.add_argument('--num_iters', type=int, default=6000, help='maximum number of iterations, 0 for unlimited') #change
parser.add_argument('--batch_size', type=int, default=100, help='batch size for parallel runs')
parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')
parser.add_argument('--log_every', type=int, default=500, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration') #change
parser.add_argument('--linf_bound', type=float, default=0, help='L_inf bound for frequency space attack')
parser.add_argument('--freq_dims', type=int, default=14, help='dimensionality of 2D frequency space')
parser.add_argument('--order', type=str, default='strided', help='(random) order of coordinate selection')
parser.add_argument('--stride', type=int, default=7, help='stride for block order')
# parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
parser.add_argument('--pixel_attack', default='False', help='attack in pixel space')
parser.add_argument('--resfile', type=str, default='./result_black_simba_diff501.json', help='suffix appended to save file')
args = parser.parse_args()
if os.path.exists(args.resfile):
    with open(args.resfile,'r') as load_f:
        res_dict = json.load(load_f)
else:
    res_dict={}

# load model and dataset
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights).cuda()
# model = resnet18(weights="IMAGENET1K_V1").cuda()
# model = getattr(models, args.model)(pretrained=True).cuda()
# model.eval()
# if args.model.startswith('inception'):
#     image_size = 299
#     testset = dset.ImageFolder(args.data_root + '/val', simba_utils.INCEPTION_TRANSFORM)
# elif:
image_size = 224
# testset = dset.ImageFolder(args.data_root, simba_utils.IMAGENET_TRANSFORM)
testset = simba_utils.ImageFolderWithID(args.data_root, simba_utils.IMAGENET_TRANSFORM)
attacker = SimBA(model, 'imagenet', image_size)
images = torch.zeros(args.num_runs, 3, image_size, image_size)
labels = torch.zeros(args.num_runs).long()
ext_imgs = torch.zeros(args.num_runs)
routes = list(range(args.num_runs))
numbers = list(range(0, len(testset) - 1))
sampled_numbers = random.sample(numbers, 49999)
sample_cnt=0
preds = labels + 1
while preds.ne(labels).sum() > 0 or ext_imgs.sum()>0:
    idx = torch.arange(0, images.size(0)).long()[torch.logical_or(preds.ne(labels),ext_imgs)]
    ext_imgs = torch.zeros(args.num_runs)
    for i in list(idx):
        if(routes[i] in res_dict.keys()):
            ext_imgs[i]=1
        res = testset[sampled_numbers[sample_cnt]]
        sample_cnt+=1
        images[i], labels[i] = res[0]
        routes[i] = res[1]
    preds[idx], _ = simba_utils.get_preds(model, images[idx], 'imagenet', batch_size=args.batch_size)
# torch.save({'images': images, 'labels': labels}, batchfile)
n_dims = 3 * image_size * image_size
if args.num_iters > 0:
    max_iters = int(min(n_dims, args.num_iters))
else:
    max_iters = int(n_dims)
N = int(math.floor(float(args.num_runs) / float(args.batch_size)))
for i in range(N):
    upper = min((i + 1) * args.batch_size, args.num_runs)
    images_batch = images[(i * args.batch_size):upper]
    labels_batch = labels[(i * args.batch_size):upper]
    
    # # replace true label with random target labels in case of targeted attack
    # if args.targeted:
    #     labels_targeted = labels_batch.clone()
    #     while labels_targeted.eq(labels_batch).sum() > 0:
    #         labels_targeted = torch.floor(1000 * torch.rand(labels_batch.size())).long()
    #     labels_batch = labels_targeted
    # for attack_iter in range(args.num_norm_iters):
    adv, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
        images_batch, labels_batch, max_iters, args.freq_dims, args.stride, args.epsilon, linf_bound=args.linf_bound,
        order=args.order, pixel_attack=args.pixel_attack, log_every=args.log_every)
    npl2_norms = l2_norms.tolist()
    nplinf_norms = linf_norms.tolist()
    npqueries = queries.tolist()
    npprobs = probs.tolist()
    for itm in range(args.batch_size):
        idx_res=itm+(i * args.batch_size)
        res_dict[routes[idx_res]]={'l2_norm':npl2_norms[itm],'linf_norm':nplinf_norms[itm],'queries:':npqueries[itm],'probs':npprobs[itm]}
    with open(args.resfile,'w') as f:
        json.dump(res_dict,f)