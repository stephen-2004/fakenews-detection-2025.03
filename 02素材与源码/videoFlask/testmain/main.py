import copy
import json
import os
import time
from tkinter import E

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import BertModel
from metrics import *
from zmq import device

from coattention import *
from layers import *

from SVFEND import SVFENDModel
import argparse
from torch.utils.data import DataLoader
import random

from testmain.utils.dataloders import SVFENDDataset









parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='SVFEND', help='SVFEND/FANVM/C3D/VGG/Bbox/Vggish/Bert/TextCNN/Comments/TikTec')
parser.add_argument('--mode_eval', default= 'cv', help='nocv/cv/temporal')
parser.add_argument('--fold', type=int,help='needed when model_eval=nocv')

parser.add_argument('--epoches', type=int, default=1)
#parser.add_argument('--batch_size', type = int, default=128)
parser.add_argument('--batch_size', type = int, default=2)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epoch_stop', type=int, default=5)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lambd', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-5)

parser.add_argument('--path_param', default= './checkpoints/')
parser.add_argument('--path_tensorboard', default= './tb/')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(2022)


#print (args)

config = {
        'model_name': args.model_name,
        'mode_eval':args.mode_eval,
        'fold':args.fold,

        'epoches': args.epoches,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epoch_stop': args.epoch_stop,
        'seed': args.seed,
        'device': args.gpu,
        'lr': args.lr,
        'lambd': args.lambd,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,

        'path_param': args.path_param,
        'path_tensorboard': args.path_tensorboard,
        }


batch_size = config['batch_size']
num_workers = config['num_workers']


def pad_sequence(seq_len,lst, emb):
    result=[]
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len=video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len,emb],dtype=torch.long)
        elif ori_len>=seq_len:
            if emb == 200:
                video=torch.FloatTensor(video[:seq_len])
            else:
                video=torch.LongTensor(video[:seq_len])
        else:
            video=torch.cat([video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.long)],dim=0)
            if emb == 200:
                video=torch.FloatTensor(video)
            else:
                video=torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def SVFEND_collate_fn(batch):
    num_comments = 23
    num_frames = 83
    num_audioframes = 50

    intro_inputid = [item['intro_inputid'] for item in batch]
    intro_mask = [item['intro_mask'] for item in batch]

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    comments_like = [item['comments_like'] for item in batch]
    comments_inputid = [item['comments_inputid'] for item in batch]
    comments_mask = [item['comments_mask'] for item in batch]

    comments_inputid_resorted = []
    comments_mask_resorted = []
    comments_like_resorted = []

    for idx in range(len(comments_like)):
        comments_like_one = comments_like[idx]
        comments_inputid_one = comments_inputid[idx]
        comments_mask_one = comments_mask[idx]
        if comments_like_one.shape != torch.Size([0]):
            comments_inputid_one, comments_mask_one, comments_like_one = (list(t) for t in zip(*sorted(
                zip(comments_inputid_one, comments_mask_one, comments_like_one), key=lambda s: s[2], reverse=True)))
        comments_inputid_resorted.append(comments_inputid_one)
        comments_mask_resorted.append(comments_mask_one)
        comments_like_resorted.append(comments_like_one)

    comments_inputid = pad_sequence(num_comments, comments_inputid_resorted, 250)
    comments_mask = pad_sequence(num_comments, comments_mask_resorted, 250)
    comments_like = []
    for idx in range(len(comments_like_resorted)):
        comments_like_resorted_one = comments_like_resorted[idx]
        if len(comments_like_resorted_one) >= num_comments:
            comments_like.append(torch.tensor(comments_like_resorted_one[:num_comments]))
        else:
            if isinstance(comments_like_resorted_one, list):
                comments_like.append(
                    torch.tensor(comments_like_resorted_one + [0] * (num_comments - len(comments_like_resorted_one))))
            else:
                comments_like.append(torch.tensor(
                    comments_like_resorted_one.tolist() + [0] * (num_comments - len(comments_like_resorted_one))))

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    audioframes = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    c3d = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    #label = [item['label'] for item in batch]

    return {
        #'label': torch.stack(label),
        'intro_inputid': torch.stack(intro_inputid),
        'intro_mask': torch.stack(intro_mask),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'comments_inputid': comments_inputid,
        'comments_mask': comments_mask,
        'comments_like': torch.stack(comments_like),
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'c3d': c3d,
        'c3d_masks': c3d_masks,
    }




def get_dataloader(data_type, data_fold,path):
    collate_fn = None

    if data_type == 'SVFEND':
        # dataset_train = SVFENDDataset(f'vid_fold_no_{data_fold}.txt')
        # dataset_test = SVFENDDataset(f'vid_fold_{data_fold}.txt')
        # dataset_train = SVFENDDataset(r"D:\竞赛\videoFlask\static\txt\video.txt")
        # dataset_test = SVFENDDataset(r"D:\竞赛\videoFlask\static\txt\video.txt")
        dataset_train = SVFENDDataset(path)
        dataset_test = SVFENDDataset(path)
        collate_fn = SVFEND_collate_fn

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=False,
                                  worker_init_fn=_init_fn,
                                  collate_fn=collate_fn)

    test_dataloader = DataLoader(dataset_test, batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 worker_init_fn=_init_fn,
                                 collate_fn=collate_fn)

    dataloaders = dict(zip(['train', 'test'], [train_dataloader, test_dataloader]))

    return dataloaders

# 按间距中的绿色按钮以运行脚本。
#if __name__ == '__main__':
def getpred(path):


    dropout=args.dropout
    model = SVFENDModel(bert_model='bert-base-chinese', fea_dim=128,
                        dropout=dropout)  # init your model class, build the graph shape
    state_dict = torch.load("checkpoints/_test_epoch1_0.8193")
    model.load_state_dict(state_dict)
    mode = 'normal'
    model_name='SVFEND'
    data_type = 'SVFEND'
    fold=1
    dataloaders=get_dataloader(data_type=data_type, data_fold=fold,path=path)

    model.cuda()
    #model.to("cpu")
    model.eval()

    pred = []
    #label = []

    if mode == "eann":
        pred_event = []
        #label_event = []

    for batch in tqdm(dataloaders['test']):
        #print(123123)
        with torch.no_grad():
            batch_data = batch
            for k, v in batch_data.items():
                 batch_data[k] = v.cuda()
                #batch_data[k] = v
            #batch_label = batch_data['label']

            if mode == "eann":
                #batch_label_event = batch_data['label_event']
                batch_outputs, batch_outputs_event, fea = model(**batch_data)
                _, batch_preds_event = torch.max(batch_outputs_event, 1)

                #label_event.extend(batch_label_event.detach().cpu().numpy().tolist())
                pred_event.extend(batch_preds_event.detach().cpu().numpy().tolist())
            else:
                batch_outputs, fea = model(**batch_data)

            _, batch_preds = torch.max(batch_outputs, 1)

            #label.extend(batch_label.detach().cpu().numpy().tolist())
            pred.extend(batch_preds.detach().cpu().numpy().tolist())
    # print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
    # print(metrics(label, pred))

    if mode == "eann" and model_name != "FANVM":
        print("event:")
        #print(accuracy_score(np.array(label_event), np.array(pred_event)))

    #print("result")
    return pred


