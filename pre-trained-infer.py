import torch
import torchvision
import torch.nn as nn
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

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
    


if __name__ == "__main__":
    acc_all = AverageMeter()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),           # 将图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    train_dataset = datasets.ImageFolder(root='/teams/dr_1765761962/program_data/binary_data_less_50', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    ###
    numClass = 2
    # default file : /root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth
    res101 = models.resnet101(pretrained=True)
    numFit = res101.fc.in_features
    res101.fc = nn.Sequential(nn.Linear(numFit, numClass), nn.Softmax(dim=1))
    res101 = res101.to(device)
    ###
    y_pred, y_gt = [], []
    y_no, y_diab = [], []
    nor, dia = AverageMeter(), AverageMeter()
    for index, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        bsz = x.shape[0]
        y_hat = res101(x)
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
            nor.update(acc_nor[0].item(), bsz)
        if len(index_dia) != 0:
            dia_element = y_hat[index_dia]
            dia_gt = y[index_dia]
            acc_dia = accuracy(dia_element, dia_gt, topk=(1,))
            dia.update(acc_dia[0].item(), bsz)
    #
    print(f'Overall acc is {acc_all.avg} normal acc is {nor.avg} diabetes acc is {dia.avg}')
    print(f'The predicted top of normal is {nor_element} ground truth is {nor_gt}')
        
