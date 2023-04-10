import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing
from skimage import measure
import torch.nn.functional as F

class Contrastive_loss(nn.Module):
    def __init__(self, margin):
        super(Contrastive_loss, self).__init__()
        self.margin = margin

    def forward(self, precision, label):
        ylen, xlen = label.shape
        num = torch.sum(label==0) + torch.sum(label==1)
        neg = torch.true_divide(torch.sum(label==0), num)
        pos = torch.true_divide(torch.sum(label==1), num)
        
        mask = torch.ones_like(label)
        mask[label==2] = 0
        
        loss = torch.mean(pos*(1-label)*torch.pow(mask*precision, 2) + \
        neg*label*torch.pow(torch.clamp(self.margin-mask*precision, min=0.0), 2))
        
        return loss
    
def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    num = res.max()
    for i in range(1, num+1):
        idy, idx = np.where(res==i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new

# def evaluate(gtImg, tstImg): 
#     m,n = gtImg.shape
    
#     TN = (gtImg==0) & (tstImg==0)
#     TN = np.sum(TN==True)

#     TP = (gtImg==255) & (tstImg==255)
#     TP = np.sum(TP==True)

#     FN = (gtImg==255) & (tstImg==0)
#     FN = np.sum(FN==True)

#     FP = (gtImg==0) & (tstImg==255)
#     FP = np.sum(FP==True)
    
#     Nc = FN + TP
#     Nu = FP + TN
#     OE = FP+FN
#     OA = (TP+TN)/(Nc+Nu)
#     Pre = TP/(TP+FP)
#     Rec = TP/(TP+FN)
#     F1 = (2*Pre*Rec)/(Pre+Rec)
#     PRA = (TP+TN)/(m*n)
#     PRE = ((TP+FP)*Nc+(FN+TN)*Nu)/(m*n)**2
#     KC = (PRA-PRE)/(1-PRE)

#     return TP,TN,FP,FN, OE,OA,Pre,Rec,F1,KC

def evaluate(gtImg, tstImg): 
    m,n = gtImg.shape
    
    TN = (gtImg<125) & (tstImg<125)
    TN = np.sum(TN==True)

    TP = (gtImg>=125) & (tstImg>=125)
    TP = np.sum(TP==True)

    FN = (gtImg>=125) & (tstImg<125)
    FN = np.sum(FN==True)

    FP = (gtImg<125) & (tstImg>=125)
    FP = np.sum(FP==True)
    
    Nc = FN + TP
    Nu = FP + TN
    OE = FP+FN
    OA = (TP+TN)/(Nc+Nu)
    Pre = TP/(TP+FP)
    Rec = TP/(TP+FN)
    F1 = (2*Pre*Rec)/(Pre+Rec)
    PRA = (TP+TN)/(m*n)
    PRE = ((TP+FP)*Nc+(FN+TN)*Nu)/(m*n)**2
    KC = (PRA-PRE)/(1-PRE)

    return TP,TN,FP,FN, OE,OA,Pre,Rec,F1,KC
