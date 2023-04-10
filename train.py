import torch
import numpy as np
from skimage import io, measure
from scipy.cluster.vq import whiten
from FCM import dicomp, hcluster
import logging
import argparse
import time
import os
from model import Model
from utils import Contrastive_loss, postprocess, evaluate
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data", default='Bern',
                    help="the name of the dataset.")

parser.add_argument("--model", default='Model',
                    help="default subtract.")

#     args = parser.parse_args()
args = parser.parse_known_args()[0]
    
if args.data == 'Bern':
    im1_path  = '/home/yyq/lds/data/SAR/newdata/Bern-SAR/im1.bmp'
    im2_path  = '/home/yyq/lds/data/SAR/newdata/Bern-SAR/im2.bmp'
    imgt_path = '/home/yyq/lds/data/SAR/newdata/Bern-SAR/im3.bmp'
    im1 = io.imread(im1_path)[:,:,0].astype(np.float32)
    im2 = io.imread(im2_path)[:,:,0].astype(np.float32)
    im_gt = io.imread(imgt_path)[:,:,0].astype(np.float32)
elif args.data == 'Ottawa':
    im1_path  = '/home/yyq/lds/data/SAR/newdata/Ottawa-SAR/im1.bmp'
    im2_path  = '/home/yyq/lds/data/SAR/newdata/Ottawa-SAR/im2.bmp'
    imgt_path = '/home/yyq/lds/data/SAR/newdata/Ottawa-SAR/im3.bmp'
    im1 = io.imread(im1_path)[:,:,0].astype(np.float32)
    im2 = io.imread(im2_path)[:,:,0].astype(np.float32)
    im_gt = io.imread(imgt_path)[:,:,0].astype(np.float32)
elif args.data == 'YellowRiverI':
    im1_path  = '/home/yyq/lds/data/SAR/newdata/YellowRiverI-SAR/im1.bmp'
    im2_path  = '/home/yyq/lds/data/SAR/newdata/YellowRiverI-SAR/im2.bmp'
    imgt_path = '/home/yyq/lds/data/SAR/newdata/YellowRiverI-SAR/referenceps.bmp'
    im1 = io.imread(im1_path).astype(np.float32)
    im2 = io.imread(im2_path).astype(np.float32)
    im_gt = io.imread(imgt_path).astype(np.float32)
elif args.data == 'YellowRiverIII':
    im1_path  = '/home/yyq/lds/data/SAR/newdata/YellowRiverIII-SAR/im1.bmp'
    im2_path  = '/home/yyq/lds/data/SAR/newdata/YellowRiverIII-SAR/im2.bmp'
    imgt_path = '/home/yyq/lds/data/SAR/newdata/YellowRiverIII-SAR/reference.bmp'
    im1 = io.imread(im1_path).astype(np.float32)
    im2 = io.imread(im2_path).astype(np.float32)
    im_gt = io.imread(imgt_path).astype(np.float32)
elif args.data == 'YellowRiverIV':
    im1_path  = '/home/yyq/lds/data/SAR/newdata/YellowRiverIV-SAR/im1.bmp'
    im2_path  = '/home/yyq/lds/data/SAR/newdata/YellowRiverIV-SAR/im2.bmp'
    imgt_path = '/home/yyq/lds/data/SAR/newdata/YellowRiverIV-SAR/im3.bmp'
    im1 = io.imread(im1_path).astype(np.float32)
    im2 = io.imread(im2_path).astype(np.float32)
    im_gt = io.imread(imgt_path).astype(np.float32)
elif args.data == 'Mexico':
    im1_path  = '/home/yyq/lds/data/SAR/newdata/Mexico/im1.bmp'
    im2_path  = '/home/yyq/lds/data/SAR/newdata/Mexico/im2.bmp'
    imgt_path = '/home/yyq/lds/data/SAR/newdata/Mexico/im3.bmp'
    im1 = io.imread(im1_path, as_gray=True).astype(np.float32)
    im2 = io.imread(im2_path, as_gray=True).astype(np.float32)
    im_gt = io.imread(imgt_path, as_gray=True).astype(np.float32)
elif args.data == 'Muragia':
    im1_path  = '/home/yyq/lds/data/SAR/newdata/Muragia/im1.bmp'
    im2_path  = '/home/yyq/lds/data/SAR/newdata/Muragia/im2.bmp'
    imgt_path = '/home/yyq/lds/data/SAR/newdata/Muragia/im3.bmp'
    im1 = io.imread(im1_path, as_gray=True).astype(np.float32)
    im2 = io.imread(im2_path, as_gray=True).astype(np.float32)
    im_gt = io.imread(imgt_path, as_gray=True).astype(np.float32)

def train(args):
    
    im_di = dicomp(im1, im2)
    ylen, xlen = im_di.shape
    pix_vec = im_di.reshape([ylen*xlen, 1])
    # pix_vec = whiten(pix_vec)
    preclassify_lab = hcluster(pix_vec, im_di)

    mdata = np.zeros([im1.shape[0], im1.shape[1], 3], dtype=np.float32)
    mdata[:,:,0] = im1
    mdata[:,:,1] = im2
    mdata[:,:,2] = im_di
    data = (torch.FloatTensor(mdata).permute(2,0,1)).unsqueeze(0)

    mlabel = preclassify_lab-1
    mlabel[mlabel==0.5] = 2
    label = torch.LongTensor(mlabel)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger(__name__)
    
    net = Model(3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    Loss = Contrastive_loss(1.6)
    
    pre_lab = torch.Tensor(mlabel)
    best_FP, best_FN, best_OE, best_OA, best_KC = 0, 0, 0, 0, 0
    train_time = 0
    
    for epoch in range(200):
        data = data.to(device)
        label = label.to(device)
        outputs = net(data)
        output = outputs.squeeze().squeeze()
        
        if epoch < 100:
            loss = Loss(output, label)
        else:
            train_start = time.time()
            
            h,w = output.shape
            out_ = output.detach().cpu().numpy()
            out_vec = out_.reshape([h*w, -1])
            out_vec = whiten(out_vec)
            label2 = hcluster(out_vec, im_di)
            label2 = label2 - 1
            label2[label2==0.5] = 2
            label2 = torch.Tensor(label2).to(device)
            loss = Loss(output, label2)

            train_end = time.time()
            epoch_train_time = train_end - train_start
            train_time += epoch_train_time

            label_1 = ~torch.le(output, 0.5)+0
            label_1 = label_1.detach().cpu().numpy()
            label_1 = postprocess(label_1)
#             TP,TN,FP,FN, OE,OA,Pre,Rec,F1,KC = evaluate(im_gt, label_1*255)
            TP,TN,FP,FN, OE,OA,Pre,Rec,F1,KC = evaluate(im_gt*255, label_1*255)
            print(TP,TN,FP,FN, OE,OA,Pre,Rec,F1,KC)
            if KC > best_KC:
                best_epoch = epoch - 1
                best_FP, best_FN, best_OE, best_OA, best_Pre, best_Rec, best_F1, best_KC = FP, FN, OE, OA, Pre, Rec, F1, KC
        #         torch.save(net.state_dict(), './checkpoint/YellowRiverIV_3')
        #         result = label_1*255
        #         result = Image.fromarray(result.astype('uint8'))
        #         result.save("./result/YellowRiverIV_3.png")

                # log
                if not os.path.exists('./log'):
                    os.makedirs('./log')
                log = open('./log/' + args.model + '+'+ args.data+'.txt', 'a')
    #             log.write('-' * 30 + '\n')
                log.write(args.data)     
                log.write('Epoch:' + str(best_epoch) + '  FP:' + str(best_FP) + \
                        '  FN:' + str(best_FN) +  '  OE:' + str(best_OE) + \
                        '  OA:' + str(best_OA) +  '  Pre:' + str(best_Pre) + \
                        '  Rec:' + str(best_Rec) +  '  F1:' + str(best_F1) + \
                        '  KC:' + str(best_KC) +'\n')
                log.write('-' * 30 + '\n')
                log.close()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #     scheduler.step()
        print('epoch',epoch, 'loss',loss)
    print('Finished Training')
    print("best_epoch", best_epoch)
    print("best_kc", best_KC)
    print("train_time", train_time)
    
def main():
    train(args)
    
if __name__ == "__main__":
    main()
