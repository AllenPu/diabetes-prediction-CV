import torch
import torchvision
import torch.nn as nn
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
 
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
 
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            #print(batch_size)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    def __init__(self,  name = '', fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

def resnet101(numClass):
    # default file : /root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth
    res101 = models.resnet101(pretrained=True)
    numFit = res101.fc.in_features
    res101.fc = nn.Sequential(nn.Linear(numFit, numClass), nn.Softmax(dim=1))
    res101 = res101.to(device)
    return res101


class LAloss(nn.Module):
    # threshold 20 : [2087, 856]; 50 : [2620,323]
    def __init__(self, threshold, cls_num_list=[], tau=1.0):
        super(LAloss, self).__init__()
        if threshold == 20:
            cls_num_list= [2087, 856]
        elif threshold == 50:
            cls_num_list= [2620,323]    
        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        iota_list = tau * np.log(cls_probs)

        self.iota_list = torch.cuda.FloatTensor(iota_list)

    def forward(self, x, target):
        #print(" x shape is {} taegt shape is {} iota is {}" .format(x.shape, target.shape, self.iota_list))
        output = x + self.iota_list

        return F.cross_entropy(output, target, reduction='sum')

    

if __name__ == "__main__":
    #
    threshold = 20
    #
    acc_all = AverageMeter()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),           # 将图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])  
    train_datasets = datasets.ImageFolder(root=f'/teams/dr_1765761962/program_data/binary_data_less_{threshold}', transform=transform)
    #
    train_size = int(0.8 * len(train_datasets))
    test_size = len(train_datasets) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_datasets, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=64, shuffle=False)
    ###
    numClass = 2
    # default file : /root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth
    model = resnet101(numClass)
    # freeze encoder but leave fc still updated
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    opt = optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
    #
    loss = nn.CrossEntropyLoss()
    #loss = LAloss(threshold)
    ###
    y_pred, y_gt = [], []
    y_no, y_diab = [], []
    nor, dia = AverageMeter(), AverageMeter()
    for e in tqdm(range(5)):
        for index, (x,y) in enumerate(train_loader):
            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            losses = loss(y_hat, y)
            losses.backward()
            opt.step()
    # test
    for index, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        bsz = x.shape[0]
        y_hat = model(x)
        acc = accuracy(y_hat, y, topk=(1,))
        acc_all.update(acc[0].item(), bsz)
        #
        index_nor = torch.nonzero(y == 0)[:,0]
        index_dia = torch.nonzero(y == 1)[:,0]
        #
        if len(index_nor) != 0:
            nor_element = y_hat[index_nor]
            nor_gt = y[index_nor]
            acc_nor = accuracy(nor_element, nor_gt, topk=(1,))
            bsz_nor = len(index_nor)
            nor.update(acc_nor[0].item(), bsz_nor)
        if len(index_dia) != 0:
            dia_element = y_hat[index_dia]
            dia_gt = y[index_dia]
            acc_dia = accuracy(dia_element, dia_gt, topk=(1,))
            bsz_dia = len(index_dia)
            dia.update(acc_dia[0].item(), bsz_dia)
    #
    print(f'Overall acc is {acc_all.avg} normal acc is {nor.avg} diabetes acc is {dia.avg}')
    print(f'The predicted top of normal is {nor_element} ground truth is {nor_gt}')
        
