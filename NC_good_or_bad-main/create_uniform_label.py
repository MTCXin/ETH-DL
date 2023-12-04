import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import pickle
import argparse

def main(args):
    
    print(args.train)
    if args.train:
        print(f"===Create training {args.type} labels===")
        trainset = CIFAR10(root="/scratch/xl998/DL/data", train=True, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]))
        the_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
    else:
        print(f"===Create test {args.type} labels===")
        testset = CIFAR10(root="/scratch/xl998/DL/data", train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]))
        the_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    
    length = args.length
    
    new_label_list = []
    for i, (data,label) in enumerate(the_loader):
        new_tar = torch.zeros(length,10)
        if args.type == "uniform":
            uniform_noise = torch.rand(length) 
            new_tar[:,label[0]] = uniform_noise / torch.linalg.norm(uniform_noise)
        elif args.type == "gaussian":
            gaussian_noise = torch.randn(length) 
            new_tar[:,label[0]] = gaussian_noise / torch.linalg.norm(gaussian_noise)
        else:
            raise ValueError("Need to specify in uniform / gaussian")
        new_label_list.append(new_tar)

    # Create pickle file containing this
    new_labels = torch.stack(new_label_list, 0)
    print(new_labels.shape)

    new_labels_p = {"label": new_labels}
    if args.train:
        with open(f"/scratch/xl998/DL/data/cifar10_{args.type}_{length}/" + 'train_label.pkl', 'wb') as f:
            pickle.dump(new_labels_p, f)
    else:
        with open(f"/scratch/xl998/DL/data/cifar10_{args.type}_{length}/" + 'test_label.pkl', 'wb') as f:
            pickle.dump(new_labels_p, f)
    print("Saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-l', '--length', type=int, required=True,
                        help='length for a label vector')
    parser.add_argument('--train', action='store_true',
                        help='Training or testing labels?')
    parser.add_argument('--type', type=str, required=True,
                        help='gaussian or uniform')
    

    args = parser.parse_args()
    print("Start")
    main(args)
