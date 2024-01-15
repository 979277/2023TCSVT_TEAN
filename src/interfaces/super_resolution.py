import string
import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
import copy
import paddle
#from src.hrnet_code.lib.models.texrnet import TexRNet
from utils import util, ssim_psnr
from IPython import embed
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
from thop import profile
from PIL import Image
import numpy as np
import torch.nn.functional as F
'''sys.path.append('../')
sys.path.append('./')'''
sys.path.append('..')
sys.path.append('.')
#from model import bicubic, srcnn, vdsr, srresnet, edsr, esrgan, rdn, lapsrn, tsrn

from src.interfaces import base
#from utils.meters import AverageMeter
from utils.metrics import get_str_list, Accuracy
from utils.util import str_filt
#from utils import utils_moran
from src.model.recognizer import resnet_aster
from src.interfaces.TextSegmentation.lib.model_zoo.get_model import get_model
from src.interfaces.TextSegmentation.lib.cfg_helper import cfg_unique_holder as cfguh, \
    get_experiment_id, \
    experiment_folder, \
    common_initiates, \
    set_debug_cfg


from src.interfaces.TextSegmentation.configs.cfg_model import cfg_texrnet as cfg_mdel
from src.interfaces.TextSegmentation.configs.cfg_base import cfg_train, cfg_test

from src.interfaces.TextSegmentation.train_utils import \
    set_cfg as set_cfg_train, \
    set_cfg_hrnetw48 as set_cfg_hrnetw48_train, \
    ts, ts_with_classifier, train

from src.interfaces.TextSegmentation.eval_utils import \
    set_cfg as set_cfg_eval, \
    set_cfg_hrnetw48 as set_cfg_hrnetw48_eval, \
    es, eval
from src.model.recognizer.tps_spatial_transformer import TPSSpatialTransformer
from src.model.recognizer.stn_head import STNHead
from src.model.recognizer.recognizer_builder import RecognizerBuilder
#from src.mmocr.mmocr.utils.ocr import MMOCR
from src.PaddleOCR.tools.infer_rec import main as SEED
from sklearn.metrics import confusion_matrix
import lmdb

#from src.interfaces.base import AsterInfo
'''from interfaces.train_utils import configuration
from src.pth.texrnet import TexRNet
from src.hrnet_code.lib.models.seg_hrnet import HighResolutionNet'''


'''cfg1=configuration()
HRNet=HighResolutionNet(cfg1.config).cuda()'''
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



class TextSR(base.TextBase):
    def writeCache(self,env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                if(isinstance(v,str)):
                    txn.put(k, v.encode())
                elif (isinstance(v,int)):
                    txn.put(k, v.encode())
                else:
                    txn.put(k, v.tobytes())
            #txn.commit()
    def create_new_dataset_train(self):
        cfg = self.config.TRAIN
        cfg1 = copy.deepcopy(cfg_test)
        cfg1.EXPERIMENT_ID = None
        cfg1.MODEL = copy.deepcopy(cfg_mdel)
        cfg1 = set_cfg_eval(cfg1, dsname='textseg')
        cfg1 = set_cfg_hrnetw48_eval(cfg1)
        cfg1.MODEL.TEXRNET.PRETRAINED_PTH = 'pth/texrnet_hrnet.pth.tar'
        #exec_es = es()
        tester = eval(cfg1)
        #tester.register_stage(exec_es)
        model1=tester(0)
        model1 =model1.to(self.device)
        model1 = torch.nn.DataParallel(model1, device_ids=range(cfg.ngpu))
        for p in model1.parameters():
            p.requires_grad = False
        model1.eval()
        #env = lmdb.open('/home/ubuntu/shurui/TextZoom-master/textzoom/train3',map_size=1099511627776)
        train_dataset, train_loader = self.get_train_data()
        c=0
        #txn = env.begin(write=True)
        for j, data in (enumerate(train_loader)):
            #import ipdb;ipdb.set_trace()
            print(j)
            images_hr, images_lr, label_strs = data
            
            if self.args.syn:
                images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor,
                                                                    self.config.TRAIN.width // self.scale_factor),
                                                        mode='bicubic')
                images_lr = images_lr.to(self.device)
            else:
                images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            img1=F.interpolate(images_hr,(4*images_hr.size(2),4*images_hr.size(3)),mode='bilinear',align_corners=True)
            #import ipdb;ipdb.set_trace()
            segmask=torch.argmax((model1(img1[ :,:3, :, :])['predrfn']),dim=1).unsqueeze(1)#单通道分割mask b,1,h,w
            imagehr_maskr=torch.zeros(images_hr.size(0),1,32,128)
            imagelr_maskr=torch.zeros(images_hr.size(0),1,16,64)
            #import ipdb;ipdb.set_trace()
            for a in range(images_hr.size(0)):
                cm0=confusion_matrix(np.array(segmask[a].cpu()).flatten(),np.array(images_hr[a][3:4,:,:].cpu()).flatten())
                cm0=torch.from_numpy(cm0)
                cm0=cm0.cuda()
                miou=torch.diag(cm0) / (torch.sum(cm0,dim=1) + torch.sum(cm0,dim=0) - torch.diag(cm0))#计算miou
                hrmean=torch.mean(segmask[a].float())*255.0#分割出的掩码平均值
                if hrmean>=35.0:#分割效果足够明显
                    if (miou[0]<=0.35)&(miou[1]<=0.35):#传统方法的分割不正确
                        imagehr_maskr=images_hr[a][3:4,:,:]
                        imagehr_maskr[torch.where(imagehr_maskr==0)]=2
                        imagehr_maskr[torch.where(imagehr_maskr==1)]=0
                        imagehr_maskr[torch.where(imagehr_maskr==2)]=1
                        images_hr[a][3:4,:,:]= imagehr_maskr#对mask做反转
                        imagelr_maskr=images_lr[a][3:4,:,:]
                        imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                        imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                        imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                        images_lr[a][3:4,:,:]= imagelr_maskr#对mask做反转
                        c=c+1
            data=images_hr, images_lr, label_strs 
        cnt = 1
        nSamples=0
        nSamples_key=b'num-samples'
        print(c)
        '''for m, data in (enumerate(train_loader)):
            print(m)
            images_hr, images_lr, label_strs = data
            if self.args.syn:
                images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor,
                                                                    self.config.TRAIN.width // self.scale_factor),
                                                        mode='bicubic')
                images_lr = images_lr.to(self.device)
            else:
                images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            cache = {}
            
            for z in range(images_hr.size(0)):
                img_HR_key = b'image_hr-%09d' % cnt  # 128*32
                img_lr_key = b'image_lr-%09d' % cnt  # 64*16
                label_key=b'label-%09d' % cnt
                cache[img_HR_key] = images_hr[z].cpu().numpy()
                cache[img_lr_key] = images_lr[z].cpu().numpy()
                cache[label_key] = label_strs[z]
                nSamples=nSamples+1
                cnt=cnt+1
                self.writeCache(env, cache)
            cache = {}
        cache[nSamples_key]=str(nSamples)
        self.writeCache(env, cache)
        env.close()'''
    
    def create_new_dataset_val(self):
        cfg = self.config.TRAIN
        cfg1 = copy.deepcopy(cfg_test)
        cfg1.EXPERIMENT_ID = None
        cfg1.MODEL = copy.deepcopy(cfg_mdel)
        cfg1 = set_cfg_eval(cfg1, dsname='textseg')
        cfg1 = set_cfg_hrnetw48_eval(cfg1)
        cfg1.MODEL.TEXRNET.PRETRAINED_PTH = 'pth/texrnet_hrnet.pth.tar'
        #exec_es = es()
        tester = eval(cfg1)
        #tester.register_stage(exec_es)
        model1=tester(0)
        model1 =model1.to(self.device)
        model1 = torch.nn.DataParallel(model1, device_ids=range(cfg.ngpu))
        for p in model1.parameters():
            p.requires_grad = False
        model1.eval()
        env = lmdb.open('/home/ubuntu/shurui/TextZoom-master/textzoom/test0/medium',map_size=3072000000)
        val_dataset_list, val_loader_list = self.get_val_data()
        #txn = env.begin(write=True)
        #for k, val_loader in enumerate(val_loader_list):
        data_name = self.config.TRAIN.VAL.val_data_dir[2].split('/')[-1]
        print('creating %s' % data_name)
        for j, data in (enumerate(val_loader_list[2])):
            print(j)
            images_hr, images_lr, label_strs = data
            if self.args.syn:
                images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor,
                                                                    self.config.TRAIN.width // self.scale_factor),
                                                        mode='bicubic')
                images_lr = images_lr.to(self.device)
            else:
                images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            img1=F.interpolate(images_hr,(4*images_hr.size(2),4*images_hr.size(3)),mode='bilinear',align_corners=True)
            segmask=torch.argmax((model1(img1[ :,:3, :, :])['predrfn']),dim=1).unsqueeze(1)#单通道分割mask b,1,h,w
            imagehr_maskr=torch.zeros(images_hr.size(0),1,32,128)
            imagelr_maskr=torch.zeros(images_hr.size(0),1,16,64)
            for a in range(images_hr.size(0)):
                cm0=confusion_matrix(np.array(segmask[a].cpu()).flatten(),np.array(images_hr[a][3:4,:,:].cpu()).flatten())
                cm0=torch.from_numpy(cm0)
                cm0=cm0.cuda()
                miou=torch.diag(cm0) / (torch.sum(cm0,dim=1) + torch.sum(cm0,dim=0) - torch.diag(cm0))#计算miou
                hrmean=torch.mean(segmask[a].float())*255.0#分割出的掩码平均值
                if hrmean>=35.0:#分割效果足够明显
                    if (miou[0]<=0.35)&(miou[1]<=0.35):#传统方法的分割不正确
                        imagehr_maskr=images_hr[a][3:4,:,:]
                        imagehr_maskr[torch.where(imagehr_maskr==0)]=2
                        imagehr_maskr[torch.where(imagehr_maskr==1)]=0
                        imagehr_maskr[torch.where(imagehr_maskr==2)]=1
                        images_hr[a][3:4,:,:]= imagehr_maskr#对mask做反转
                        imagelr_maskr=images_lr[a][3:4,:,:]
                        imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                        imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                        imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                        images_lr[a][3:4,:,:]= imagelr_maskr#对mask做反转
            data=images_hr, images_lr, label_strs 
        #for k, val_loader in enumerate(val_loader_list):
        data_name = self.config.TRAIN.VAL.val_data_dir[2].split('/')[-1]
        print('creating %s' % data_name)
        nSamples=0
        cnt = 1
        nSamples_key=b'num-samples'
        for m, data in (enumerate(val_loader_list[2])):
            print(m)
            images_hr, images_lr, label_strs = data
            #import ipdb;ipdb.set_trace()
            if self.args.syn:
                images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor,
                                                                    self.config.TRAIN.width // self.scale_factor),
                                                        mode='bicubic')
                images_lr = images_lr.to(self.device)
            else:
                images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            cache = {}
            
            for z in range(images_hr.size(0)):
                img_HR_key = b'image_hr-%09d' % cnt  # 128*32
                img_lr_key = b'image_lr-%09d' % cnt  # 64*16
                label_key=b'label-%09d' % cnt
                cache[img_HR_key] = images_hr[z].cpu().numpy()
                cache[img_lr_key] = images_lr[z].cpu().numpy()
                cache[label_key] = label_strs[z]
                nSamples=nSamples+1
                cnt=cnt+1
                self.writeCache(env, cache)
            cache = {}
        cache[nSamples_key]=str(nSamples)
        self.writeCache(env, cache)
        env.close()


    def train(self):
        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data0()
        val_dataset_list, val_loader_list = self.get_val_data0()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        loss1,loss2=model_dict['L1'],model_dict['KLD']
        lossmse=nn.MSELoss()
        '''cfg1 = copy.deepcopy(cfg_test)
        cfg1.EXPERIMENT_ID = None
        cfg1.MODEL = copy.deepcopy(cfg_mdel)
        cfg1 = set_cfg_eval(cfg1, dsname='textseg')
        cfg1 = set_cfg_hrnetw48_eval(cfg1)
        cfg1.MODEL.TEXRNET.PRETRAINED_PTH = 'pth/texrnet_hrnet.pth.tar'
        #exec_es = es()
        tester = eval(cfg1)
        #tester.register_stage(exec_es)
        model1=tester(0)
        model1 =model1.to(self.device)
        model1 = torch.nn.DataParallel(model1, device_ids=range(cfg.ngpu))
        for p in model1.parameters():
            p.requires_grad = False
        model1.eval()
        for m in model1.parameters():
            m.requires_grad=False'''
        
        
        # 初始化
        '''ema = EMA(model, 0.999)
        ema.register()'''
        aster, aster_info = self.Aster_init()
        
        astertp0=RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True).cuda()
        astertp0.load_state_dict(torch.load('/home/tongji/shurui/TextZoom-master/src/pth/demo.pth.tar')['state_dict'])
        astertp=astertp0.encoder
        astertp = astertp.to(self.device)
        astertp = torch.nn.DataParallel(astertp, device_ids=range(cfg.ngpu))
        optimizer_G = self.optimizer_init(model)
        
        


        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []
        for epoch in range(cfg.epochs):
            import ipdb;ipdb.set_trace()
            for j, data in (enumerate(train_loader)):
                for p in model.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j + 1
                images_hr, images_lr, label_strs = data
                if self.args.syn:
                    images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor,
                                                                      self.config.TRAIN.width // self.scale_factor),
                                                          mode='bicubic')
                    images_lr = images_lr.to(self.device)
                else:
                    images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                for z in range(self.batch_size):
                    lrmaskrow=torch.mean(torch.cat((images_lr[z][3:4,0:0,:],images_lr[z][3:4,-1:,:]),1))
                    lrmaskcolumn=torch.mean(torch.cat((images_lr[z][3:4,:,0:0],images_lr[z][3:4,:,-1:]),2))
                    lrmean=1/2*(lrmaskcolumn+lrmaskrow)*255.0
                    hrmaskrow=torch.mean(torch.cat((images_hr[z][3:4,0:0,:],images_hr[z][3:4,-1:,:]),1))
                    hrmaskcolumn=torch.mean(torch.cat((images_hr[z][3:4,:,0:0],images_hr[z][3:4,:,-1:]),2))
                    hrmean=1/2*(hrmaskcolumn+hrmaskrow)*255.0
                    if((lrmean>190)|(hrmean>190)):
                        imagehr_maskr=images_hr[z][3:4,:,:]
                        imagehr_maskr[torch.where(imagehr_maskr==0)]=2
                        imagehr_maskr[torch.where(imagehr_maskr==1)]=0
                        imagehr_maskr[torch.where(imagehr_maskr==2)]=1
                        images_hr[z][3:4,:,:]= imagehr_maskr#对mask做反转
                        imagelr_maskr=images_lr[z][3:4,:,:]
                        imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                        imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                        imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                        images_lr[z][3:4,:,:]= imagelr_maskr#对mask做反转
                model.train()
                image_sr2x,image_sr4x,att_inf_avg=model(images_lr)
                #image_sr2x,image_sr4x,att_inf_avg=model(images_lr)
                #image_sr2x,image_sr4x,att_inf_avg=model(images_lr)
                #image_sr2x,att_inf_avg=model(images_lr)
                #image_sr2x,att_inf_avg=model(images_lr)
                #image_sr2x,image_sr4x,image_sr8x,att_inf_avg=model(images_lr)
                tp2=astertp(self.parse_aster_data(image_sr2x[:, :3, :, :])['images'])
                image_sr4x0=F.interpolate(image_sr4x,(images_hr.size(2),images_hr.size(3)),mode='bicubic',align_corners=True)
                #image_sr8x0=F.interpolate(image_sr8x,(images_hr.size(2),images_hr.size(3)),mode='bicubic',align_corners=True)
                tp=astertp(self.parse_aster_data(images_hr[:, :3, :, :])['images'])
                tp1=astertp(self.parse_aster_data(image_sr4x0[:, :3, :, :])['images'])
                #tp3=astertp(self.parse_aster_data(image_sr8x0[:, :3, :, :])['images'])
                tp1_input=torch.log_softmax(tp1,dim=1)
                tp2_input=torch.log_softmax(tp2,dim=1)
                tp_input=torch.softmax(tp,dim=1)
                #tp3_input=torch.softmax(tp3,dim=1)
                L11=loss1(tp1,tp.detach().data).mean()
                L12=loss2(tp1_input,tp_input.detach().data).mean()
                L21=loss1(tp2,tp.detach().data).mean()
                L22=loss2(tp2_input,tp_input.detach().data).mean()
                '''L31=loss1(tp3,tp.detach().data).mean()
                L32=loss2(tp3_input,tp_input.detach().data).mean()'''
                #att_inf_avg0=F.interpolate(att_inf_avg0,(32,128),mode='bilinear',align_corners=False)
                #Lmask=2/3*(lossmse(att_inf_avg,images_hr[:, 3:4, :, :].detach().data).mean())+1/3*(lossmse(att_inf_avg0,images_hr[:, 3:4, :, :].detach().data).mean())
                Lmask=lossmse(att_inf_avg,images_hr[:, 3:4, :, :].detach().data).mean()#+lossmse(att_inf,images_hr[:, 3:4, :, :].detach().data).mean())
                #Lts=2/3*(L11+L12)+1/3*(L21+L22)
                #Lts=(L21+L22)
                #loss_im = (image_crit(image_sr2x, images_hr).mean() * 100+L21+L22)+Lmask
                
                Lts=2/3*(L11+L12)+1/3*(L21+L22)
                loss_im = 2/3*(image_crit(image_sr4x0, images_hr).mean() * 100+L11+L12)+1/3*(image_crit(image_sr2x, images_hr).mean() * 100+L21+L22)+Lmask
                #loss_im = (image_crit(image_sr2x, images_hr).mean() * 100+L21+L22)+Lmask
                #loss_im = 1/2*(image_crit(image_sr8x0, images_hr).mean() * 100+L31+L32)+1/4*(image_crit(image_sr2x, images_hr).mean() * 100+L21+L22)+Lmask+1/4*(image_crit(image_sr4x0, images_hr).mean() * 100+L11+L12)

                torch.autograd.set_detect_anomaly(True)
                optimizer_G.zero_grad()
                loss_im.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()
                # 训练过程中，更新完参数后，同步update shadow weights
                #ema.update()
                if iters % cfg.displayInterval == 0:
                    print('[{}]\t'
                          'Epoch: [{}][{}/{}]\t'
                          'vis_dir={:s}\t'
                          '{:.3f} \t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  self.vis_dir,
                                  float(loss_im.data)))
                    f = open('log.txt', 'a')
                    '''f.write('[{}]\t'
                          'Epoch: [{}][{}/{}]\t'
                          'vis_dir={:s}\t'
                          '{:.3f} \t'
                          'Ltsm={:.3f}\t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  self.vis_dir,
                                  float(loss_im.data),
                                  float(Lmask.data)))'''
                    f.write('[{}]\t'
                          'Epoch: [{}][{}/{}]\t'
                          'vis_dir={:s}\t'
                          '{:.3f} \t'
                          
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  self.vis_dir,
                                  float(loss_im.data),
                                  ))
                    f.write('\n')
                    f.close()
                if iters % cfg.VAL.valInterval == 0:
                    print('======================================================')
                    f = open('log.txt', 'a')
                    f.write('======================================================')
                    f.write('\n')
                    f.close()
                    current_acc_dict = {}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        f = open('log.txt', 'a')
                        print('evaling %s' % data_name)
                        f.write('evaling %s' % data_name)
                        f.close()
                        metrics_dict = self.eval1(model, val_loader, image_crit, iters, aster, aster_info)#,astertp
                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))
                            f = open('log.txt', 'a')
                            f.write('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))
                            f.write('\n')
                            f.close()

                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                            f = open('log.txt', 'a')
                            f.write('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                            f.write('\n')
                            f.close()
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model')
                        f = open('log.txt', 'a')
                        f.write('saving best model')
                        f.write('\n')
                        f.close()
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list)
        
    def eval1(self, model, val_loader, image_crit, index, aster, aster_info):#,astertp
        for p in model.parameters():
            p.requires_grad = False
        for p in aster.parameters():
            p.requires_grad = False
        
        # eval前，apply shadow weights；eval之后，恢复原来模型的参数
        '''ema = EMA(model, 0.999)
        ema.register()
        ema.apply_shadow()
        
        cfg=self.config.TRAIN'''
        '''astertp0=RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True).cuda()
        astertp0.load_state_dict(torch.load('/home/ubuntu/shurui/TextZoom-master/src/pth/demo.pth.tar')['state_dict'])
        astertp=astertp0.encoder
        astertp = astertp.to(self.device)
        astertp = torch.nn.DataParallel(astertp, device_ids=range(cfg.ngpu))'''
    
        '''crnn = self.CRNN_init()
        crnn.eval()'''
        '''moran = self.MORAN_init()
        moran.eval()'''
        model.eval()
        aster.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        with torch.no_grad():
            for i, data in (enumerate(val_loader)):
                images_hr, images_lr, label_strs = data
                val_batch_size = images_lr.shape[0]
                
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                for z in range(self.batch_size):
                    lrmaskrow=torch.mean(torch.cat((images_lr[z][3:4,0:0,:],images_lr[z][3:4,-1:,:]),1))
                    lrmaskcolumn=torch.mean(torch.cat((images_lr[z][3:4,:,0:0],images_lr[z][3:4,:,-1:]),2))
                    lrmean=1/2*(lrmaskcolumn+lrmaskrow)*255.0
                    hrmaskrow=torch.mean(torch.cat((images_hr[z][3:4,0:0,:],images_hr[z][3:4,-1:,:]),1))
                    hrmaskcolumn=torch.mean(torch.cat((images_hr[z][3:4,:,0:0],images_hr[z][3:4,:,-1:]),2))
                    hrmean=1/2*(hrmaskcolumn+hrmaskrow)*255.0
                    if((lrmean>190)|(hrmean>190)):
                        imagehr_maskr=images_hr[z][3:4,:,:]
                        imagehr_maskr[torch.where(imagehr_maskr==0)]=2
                        imagehr_maskr[torch.where(imagehr_maskr==1)]=0
                        imagehr_maskr[torch.where(imagehr_maskr==2)]=1
                        images_hr[z][3:4,:,:]= imagehr_maskr#对mask做反转
                        imagelr_maskr=images_lr[z][3:4,:,:]
                        imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                        imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                        imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                        images_lr[z][3:4,:,:]= imagelr_maskr#对mask做反转
                #image_sr2x,_=model(images_lr)
                image_sr2x,image_sr4x,att_inf_avg=model(images_lr)
                #image_sr2x,image_sr4x,image_sr8x,att_inf_avg=model(images_lr)
               
                
                image_sr4x1=F.interpolate(image_sr4x,(images_hr.size(2),images_hr.size(3)),mode='bicubic',align_corners=True)
                #image_sr8x1=F.interpolate(image_sr8x,(images_hr.size(2),images_hr.size(3)),mode='bicubic',align_corners=True)
                    
                metric_dict['psnr'].append(self.cal_psnr(image_sr4x1, images_hr))
                metric_dict['ssim'].append(self.cal_ssim(image_sr4x1, images_hr))
                aster_dict_sr = self.parse_aster_data(image_sr4x[:, :3, :, :])
                #aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                #aster_output_lr = aster(aster_dict_lr)
                aster_output_sr = aster(aster_dict_sr)
                #pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                #pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
                '''moran_input = self.parse_moran_data(image_sr4x[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]'''
                '''crnn_input_sr = self.parse_crnn_data(image_sr4x[:, :3, :, :])
                #crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output_sr = crnn(crnn_input_sr)
                #crnn_output_lr = crnn(crnn_input_lr)
                _, preds_sr = crnn_output_sr.max(2)
                #_, preds_lr = crnn_output_lr.max(2)
                preds_sr = preds_sr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_sr.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds_sr.data, preds_size.data, raw=False)'''
                for pred, target in zip(pred_str_sr, label_strs):
                    if pred == str_filt(target, 'lower'):
                        n_correct += 1

                #loss_im = 1/3*(image_crit(image_sr2x,images_hr).mean()*100)+2/3*(image_crit(image_sr4x1,images_hr).mean()*100)
                loss_im = 1/3*(image_crit(image_sr2x,images_hr).mean()*100)
                #loss_rec = aster_output_sr['losses']['loss_rec'].mean()
                sum_images += val_batch_size
                torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        f = open('log.txt', 'a')
        '''print('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(loss_rec.data), float(loss_im.data),
                      float(psnr_avg), float(ssim_avg), ))'''
        print('[{}]\t'
              'loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(loss_im.data),
                      float(psnr_avg), float(ssim_avg), ))
        print('save display images')
        #self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)
        accuracy = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        print('aster_accuray: %.2f%%' % (accuracy * 100))
        '''f.write('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(loss_rec.data), float(loss_im.data),
                      float(psnr_avg), float(ssim_avg), ))'''
        f.write('[{}]\t'
              'loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(loss_im.data),
                      float(psnr_avg), float(ssim_avg), ))
        f.write('\n')
        f.write('save display images')
        f.write('\n')
        f.write('aster_accuray: %.2f%%' % (accuracy * 100))
        f.write('\n')
        f.close()
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg
        return metric_dict

    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        test_data, test_loader = self.get_test_data0(self.test_data_dir)
        data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        # print(sum(p.numel() for p in moran.parameters()))
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        sr_time = 0
        lr_correct=0
        sr_correct=0
        for i, data in (enumerate(test_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            for z in range(val_batch_size):
                    lrmaskrow=torch.mean(torch.cat((images_lr[z][3:4,0:0,:],images_lr[z][3:4,-1:,:]),1))
                    lrmaskcolumn=torch.mean(torch.cat((images_lr[z][3:4,:,0:0],images_lr[z][3:4,:,-1:]),2))
                    lrmean=1/2*(lrmaskcolumn+lrmaskrow)*255.0
                    hrmaskrow=torch.mean(torch.cat((images_hr[z][3:4,0:0,:],images_hr[z][3:4,-1:,:]),1))
                    hrmaskcolumn=torch.mean(torch.cat((images_hr[z][3:4,:,0:0],images_hr[z][3:4,:,-1:]),2))
                    hrmean=1/2*(hrmaskcolumn+hrmaskrow)*255.0
                    if((lrmean>190)|(hrmean>190)):
                        imagehr_maskr=images_hr[z][3:4,:,:]
                        imagehr_maskr[torch.where(imagehr_maskr==0)]=2
                        imagehr_maskr[torch.where(imagehr_maskr==1)]=0
                        imagehr_maskr[torch.where(imagehr_maskr==2)]=1
                        images_hr[z][3:4,:,:]= imagehr_maskr#对mask做反转
                        imagelr_maskr=images_lr[z][3:4,:,:]
                        imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                        imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                        imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                        images_lr[z][3:4,:,:]= imagelr_maskr#对mask做反转
            sr_beigin = time.time()
            images_sr2x,images_sr4x,_ = model(images_lr)
            images_sr4x0=F.interpolate(images_sr4x,(images_hr.size(2),images_hr.size(3)),mode='bicubic',align_corners=True)
            #images_sr2x,_ = model(images_lr)

            # images_sr = images_lr
            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            '''metric_dict['psnr'].append(self.cal_psnr(images_sr4x0, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr4x0, images_hr))'''
            metric_dict['psnr'].append(self.cal_psnr(images_sr4x0, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr4x0, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr4x[:, :3, :, :])
                #import ipdb;ipdb.set_trace()
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr4x[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr4x[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            elif self.args.rec == 'nrtr':
                ocr=MMOCR(recog='NRTR_1/8-1/4')
                for k in range(val_batch_size):
                    result_list_lr=ocr.readtext(images_lr[k][:3,:,:], print_result=False, imshow=False)
                    pred_str_lr=result_list_lr[0]['text']
                    #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    result_list_sr=ocr.readtext(images_sr4x[k][:3,:,:], print_result=False, imshow=False)
                    pred_str_sr=result_list_sr[0]['text']
                    sum_images=sum_images+1
                    if(str_filt(pred_str_lr, 'lower')==str_filt(label_strs[k],'lower')):
                        lr_correct=lr_correct+1
                    if(str_filt(pred_str_sr, 'lower')==str_filt(label_strs[k],'lower')):
                        sr_correct=sr_correct+1
            #import ipdb;ipdb.set_trace()
            '''for pred, target in zip(pred_str_sr, label_strs):
                if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                    n_correct += 1
            sum_images += val_batch_size'''
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{}/{}]\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader), ))
            #self.test_display(images_lr[:,:3,:,:], images_sr4x0[:,:3,:,:], images_hr[:,:3,:,:], pred_str_lr, pred_str_sr, label_strs, str_filt)
        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        fps = sum_images/(time_end - time_begin)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        acc_lr=round(lr_correct / sum_images, 4)
        acc_sr=round(sr_correct / sum_images, 4)
        # result = {'accuracy': current_acc_dict, 'fps': fps}
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps,'acc_lr':acc_lr,'acc_sr':acc_sr}
        print(result)
    def seg(self):
        cfg=self.config.TRAIN
        #train_dataset, train_loader = self.get_train_data0()
        '''model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']'''
        test_data, test_loader = self.get_test_data(self.test_data_dir)
        data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)
        '''cfg1 = copy.deepcopy(cfg_test)
        cfg1.EXPERIMENT_ID = None
        cfg1.MODEL = copy.deepcopy(cfg_mdel)
        cfg1 = set_cfg_eval(cfg1, dsname='textseg')
        cfg1 = set_cfg_hrnetw48_eval(cfg1)
        cfg1.MODEL.TEXRNET.PRETRAINED_PTH = 'pth/texrnet_hrnet.pth.tar'
        exec_es = es()
        tester = eval(cfg1)
        
        tester.register_stage(exec_es)

        
        model=tester(0)'''


        
        #model.load_state_dict(torch.load(self.config.TRAIN.VAL.texrnet_pretrained))#['state_dict']
        
        
        #torch.save(model.state_dict(), 'pth/texrnet_hrnet.pth.tar', _use_new_zipfile_serialization=False)
        #model.load_state_dict(torch.load('pth/texrnet_hrnet.pth.tar',map_location='cpu'))
        '''model =model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=range(cfg.ngpu))'''
        #import ipdb;ipdb.set_trace()
        '''model.eval()

        for p in model.parameters():
            p.requires_grad = False'''
        #loss1=nn.L1Loss()
        '''aster, aster_info = self.Aster_init()
        aster.eval()'''
        #optimizer_G = self.optimizer_init(model)
        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        index=0
        
        images_lr1=torch.ones(1,32,128)
        images_hr1=torch.ones(1,32,128)
        for i, data in (enumerate(test_loader)):
            
            images_hr, images_lr, label_strs = data
            images_lr=F.interpolate(images_lr,(32,128),mode='bicubic')
            images_hr=F.interpolate(images_hr,(32,128),mode='bicubic')
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            for z in range(val_batch_size):
                    
                    lrmaskrow=torch.mean(torch.cat((images_lr[z][3:4,0:0,:],images_lr[z][3:4,-1:,:]),1))
                    lrmaskcolumn=torch.mean(torch.cat((images_lr[z][3:4,:,0:0],images_lr[z][3:4,:,-1:]),2))
                    lrmean=1/2*(lrmaskcolumn+lrmaskrow)*255.0
                    hrmaskrow=torch.mean(torch.cat((images_hr[z][3:4,0:0,:],images_hr[z][3:4,-1:,:]),1))
                    hrmaskcolumn=torch.mean(torch.cat((images_hr[z][3:4,:,0:0],images_hr[z][3:4,:,-1:]),2))
                    hrmean=1/2*(hrmaskcolumn+hrmaskrow)*255.0
                    if((lrmean>190)|(hrmean>190)):
                        print('save display images')
                        images_lr1=images_lr[z][3:4,:,:]
                        images_hr1=images_hr[z][3:4,:,:]
                        before=1
                        self.tripple_display3(images_lr[z][3:4,:,:],images_hr[z][3:4,:,:],images_lr[z],images_hr[z],label_strs[z],index,before)
                        imagehr_maskr=images_hr[z][3:4,:,:]
                        imagehr_maskr[torch.where(imagehr_maskr==0)]=2
                        imagehr_maskr[torch.where(imagehr_maskr==1)]=0
                        imagehr_maskr[torch.where(imagehr_maskr==2)]=1
                        images_hr[z][3:4,:,:]= imagehr_maskr#对mask做反转
                        imagelr_maskr=images_lr[z][3:4,:,:]
                        imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                        imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                        imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                        images_lr[z][3:4,:,:]= imagelr_maskr#对mask做反转
                        before=0
                        self.tripple_display3(images_lr[z][3:4,:,:],images_hr[z][3:4,:,:],images_lr[z],images_hr[z],label_strs[z],index,before)
                        index=index+1
            sr_beigin = time.time()
            #images_sr2x,avg_mask = model(images_lr)
            #images_sr2x,images_sr4x,avg_mask0,avg_mask = model(images_lr)
            #images_sr4x0=F.interpolate(images_sr4x,(images_hr.size(2),images_hr.size(3)),mode='bicubic',align_corners=True)
            # images_sr = images_lr
            #avg_mask0=F.interpolate(avg_mask0,(32,128),mode='bilinear',align_corners=False)
            '''images_sr1=(torch.argmax(model(images_lr4[ :,:3, :, :])['predrfn'],dim=1).unsqueeze(1).float())*images_lr3
            images_hr1=(torch.argmax(model(images_hr4[ :,:3, :, :])['predrfn'],dim=1).unsqueeze(1).float())*images_hr'''
            transform0 = transforms.Compose(
            [transforms.ToPILImage()]
            )
            transform1=transforms.Compose(
                [transforms.ToTensor()]
            )
            
            
            
            


    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            #img = img.resize((256, 32), Image.BICUBIC)
            #img = img.resize((64, 16), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                lrmaskrow=torch.mean(torch.cat((mask[:,0:0,:],mask[:,-1:,:]),1))
                lrmaskcolumn=torch.mean(torch.cat((mask[:,:,0:0],mask[:,:,-1:]),2))
                lrmean=1/2*(lrmaskcolumn+lrmaskrow)*255.0
                
                if((lrmean>190)):
                    imagelr_maskr=mask
                    imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                    imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                    imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                    mask= imagelr_maskr#对mask做反转
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor
        def transformlr_(path):
            img = Image.open(path)
            #img = img.resize((256, 32), Image.BICUBIC)
            #img = img.resize((64, 16), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            '''if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                lrmaskrow=torch.mean(torch.cat((mask[:,0:0,:],mask[:,-1:,:]),1))
                lrmaskcolumn=torch.mean(torch.cat((mask[:,:,0:0],mask[:,:,-1:]),2))
                lrmean=1/2*(lrmaskcolumn+lrmaskrow)*255.0
                
                if((lrmean>190)):
                    imagelr_maskr=mask
                    imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                    imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                    imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                    mask= imagelr_maskr#对mask做反转
                img_tensor = torch.cat((img_tensor, mask), 0)'''
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        elif self.args.rec=='autostr':
            autostr,autostr_info=self.AutoSTR_init()
            autostr.eval()
        elif self.args.rec=='seed':
            SEED_rec,post_process_class,device=SEED()
            #device0=paddle.CPUPlace()
            #paddle.device.set_device("gpu")
            #import ipdb;ipdb.set_trace()
            #SEED_rec=SEED_rec.to(device0)
            #model=model.to(self.device)
            with torch.no_grad():
                SEED_rec.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        j=0
        path_list=os.listdir(self.args.demo_dir)
        sr_count=0
        sr_correct=0
        lr_correct=0
        #import ipdb;ipdb.set_trace()
 
        path_list.sort(key=lambda x : int(x.split('.')[0].split('_')[1])) 
        #import ipdb;ipdb.set_trace()
        for im_name in tqdm(path_list):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr0 = transformlr_(os.path.join(self.args.demo_dir, im_name))
            images_lr0 = images_lr0.to(self.device)
            #images_lr = transforms.ToTensor()(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
            #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
            sr_count=sr_count+1
            images_sr0,images_sr,_ = model(images_lr)
            #images_sr= model(images_lr)
            fo1=open("/home/ubuntu/sr/TextZoom-master/gt_test.txt","r")
            lines2 = [l[l.rfind(','):] for l in fo1.readlines() if l.strip()]
            label=lines2[j]
            for i in label:
                if((i==',')|(i=='\n')|(i==' ')|(i=='"')):
                    label=label.replace(i,'')
            #import ipdb;ipdb.set_trace()

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                '''moran_input = self.parse_moran_data(images_lr[:, :3, :, :])
                #import ipdb;ipdb.set_trace()
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                #import ipdb;ipdb.set_trace()
                pred_str_lr = sim_preds.split('$')[0]'''
                #pred_str_lr = [pred.split('$')[0] for pred in sim_preds]
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                    moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                        debug=True)
                    preds, preds_reverse = moran_output[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                    pred_str_sr = sim_preds.split('$')[0]
                #pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    moran_input_lr = self.parse_moran_data(images_lr0[:, :3, :, :])
                    moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3], test=True,
                                        debug=True)
                    preds_lr, preds_reverse_lr = moran_output_lr[0]
                    _, preds_lr = preds_lr.max(1)
                    sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                    #pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
                    pred_str_lr = sim_preds_lr.split('$')[0]
                    #import ipdb;ipdb.set_trace()
            elif self.args.rec == 'aster':
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                    aster_output_sr = aster(aster_dict_sr)
                    pred_rec_sr = aster_output_sr['output']['pred_rec']
                    pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
               

                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    aster_dict_lr = self.parse_aster_data(images_lr0[:, :3, :, :])
                    aster_output_lr = aster(aster_dict_lr)
                    pred_rec_lr = aster_output_lr['output']['pred_rec']
                    pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                    crnn_output = crnn(crnn_input)
                    _, preds = crnn_output.max(2)
                    preds = preds.transpose(1, 0).contiguous().view(-1)
                    preds_size = torch.IntTensor([crnn_output.size(0)] * 1)
                    pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    crnn_input_lr = self.parse_crnn_data(images_lr0[:, :3, :, :])
                    crnn_output_lr = crnn(crnn_input_lr)
                    _, preds_lr = crnn_output_lr.max(2)
                    preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                    preds_size = torch.IntTensor([crnn_output_lr.size(0)] * 1)
                    pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            elif self.args.rec=='autostr':
                '''autostr_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                autostr_output_sr = aster(autostr_dict_sr)
                pred_rec_sr = autostr_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, autostr_dict_sr['rec_targets'], dataset=autostr_info)
               '''

                autostr_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                autostr_output_lr = autostr(autostr_dict_lr)
                pred_rec_lr = autostr_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, autostr_dict_lr['rec_targets'], dataset=autostr_info)
            elif self.args.rec=='nrtr':
                #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                ocr=MMOCR(recog='NRTR_1/8-1/4')
                result_list_lr=ocr.readtext(images_lr[:,:3,:,:], print_result=False, imshow=False)
                pred_str_lr=result_list_lr[0]['text']
                #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                result_list_sr=ocr.readtext(images_sr[:,:3,:,:], print_result=False, imshow=False)
                pred_str_sr=result_list_sr[0]['text']
            elif self.args.rec=='master':
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    ocr=MMOCR(recog='MASTER')
                    result_list_lr=ocr.readtext(images_lr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_lr=result_list_lr[0]['text']
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    result_list_sr=ocr.readtext(images_sr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_sr=result_list_sr[0]['text']
            elif self.args.rec=='abinet':
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    ocr=MMOCR(recog='ABINet')
                    result_list_lr=ocr.readtext(images_lr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_lr=result_list_lr[0]['text']
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    result_list_sr=ocr.readtext(images_sr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_sr=result_list_sr[0]['text']
            elif self.args.rec=='sar':
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    ocr=MMOCR(recog='SAR')
                    result_list_lr=ocr.readtext(images_lr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_lr=result_list_lr[0]['text']
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    result_list_sr=ocr.readtext(images_sr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_sr=result_list_sr[0]['text']
            elif self.args.rec=='satrn':
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    ocr=MMOCR(recog='SATRN')
                    result_list_lr=ocr.readtext(images_lr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_lr=result_list_lr[0]['text']
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    result_list_sr=ocr.readtext(images_sr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_sr=result_list_sr[0]['text']
            elif self.args.rec=='robust':
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    ocr=MMOCR(recog='RobustScanner')
                    result_list_lr=ocr.readtext(images_lr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_lr=result_list_lr[0]['text']
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                    result_list_sr=ocr.readtext(images_sr[:,:3,:,:], print_result=False, imshow=False)
                    pred_str_sr=result_list_sr[0]['text']
            elif self.args.rec=='seed':
                if(((images_lr0.size(2)*images_lr0.size(3))<(64*16))):
                    #import ipdb;ipdb.set_trace()
                    b,c,h,w=images_lr.size()
                    #import ipdb;ipdb.set_trace()
                    images_lr=list(images_lr.cpu().numpy().reshape(c,h,w))
                    images_lr = np.expand_dims(images_lr, axis=0)
                    images_lr = paddle.to_tensor(images_lr)
                    #images_lr=images_lr.cpu()
                    #import ipdb;ipdb.set_trace()
                    with torch.no_grad():
                        pred_lr=SEED_rec(images_lr[:,:3,:,:])
                        pred_str_lr = post_process_class(pred_lr)[0][0]
                    #info = None
                    #pred_str_lr=result_list_lr[0]['text']
                if(((images_lr0.size(2)*images_lr0.size(3))<(64*16))):
                    b0,c0,h0,w0=images_sr.size()
                    #import ipdb;ipdb.set_trace()
                    images_sr=list(images_sr.cpu().numpy().reshape(c0,h0,w0))
                    #images_sr=list(images_sr.squeeze().cpu().numpy())
                    images_sr = np.expand_dims(images_sr, axis=0)
                    images_sr = paddle.to_tensor(images_sr)
                    #images_sr=images_sr.cpu()
                    with torch.no_grad():
                        pred_sr=SEED_rec(images_sr[:,:3,:,:])
                        pred_str_sr = post_process_class(pred_sr)[0][0]
                #import ipdb;ipdb.set_trace()
            j=j+1
            #import ipdb;ipdb.set_trace()
            
            #self.test_display_IC15(images_lr[:,:3,:,:],images_sr[:,:3,:,:],pred_str_lr,pred_str_sr,label,str_filt)
            #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
            '''if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                if(len(pred_str_lr)>0):
                    if(str_filt(pred_str_lr[0], 'lower') == str_filt(label, 'lower')):
                            lr_correct=lr_correct+1
            if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
            #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                if(len(pred_str_sr)>0):
                    if(str_filt(pred_str_sr[0], 'lower') == str_filt(label, 'lower')):
                            sr_correct=sr_correct+1'''
            '''if(len(pred_str_lr)>0):
                if(str_filt(pred_str_lr[0], 'lower') == str_filt(label, 'lower')):
                    n_correct=n_correct+1'''
            '''if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                if(str_filt(pred_str_lr, 'lower') == str_filt(label, 'lower')):
                        lr_correct=lr_correct+1
            if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                if(str_filt(pred_str_sr, 'lower') == str_filt(label, 'lower')):
                        sr_correct=sr_correct+1'''
            #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
            if(((images_lr0.size(2)*images_lr0.size(3))<(64*16))):
                if(str_filt(pred_str_lr, 'lower') == str_filt(label, 'lower')):
                        lr_correct=lr_correct+1
            #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
            if(((images_lr0.size(2)*images_lr0.size(3))<(64*16))):
                if(str_filt(pred_str_sr, 'lower') == str_filt(label, 'lower')):
                        sr_correct=sr_correct+1
            #print(pred_str_sr[0], '===>', label)
            torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        accuracy_sr=round(sr_correct / sr_count, 4)
        accuracy_lr=round(lr_correct / sr_count, 4)
        #accuracy=round(n_correct / sum_images, 4)
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        print('fps=', fps)
        print('sr_count=', sr_count)
        print('accuracy_lr=', accuracy_lr)
        print('accuracy_sr=', accuracy_sr)
        #print('accuracy=',accuracy)
    def test_regular(self):
        mask_ = self.args.mask

        def transform_(img,img_gray):
            '''img = Image.open(path)
            #img = img.resize((256, 32), Image.BICUBIC)
            #img = img.resize((64, 16), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)'''
            if mask_:
                import ipdb;ipdb.set_trace()
                mask0=img.numpy()
                for i in range(img.size(0)):
                    
                    mask_0=Image.fromarray(mask0[i][0:1,:,:])
                    mask_1=Image.fromarray(mask0[i][1:2,:,:])
                    mask_2=Image.fromarray(mask0[i][2:3,:,:])
                    mask=torch.cat((mask_0,mask_1,mask_2),1)
                    mask=mask.convert('L')
                    import ipdb;ipdb.set_trace()
                    thres = torch.mean(mask)
                    
                    #mask=np.array(mask)
                    mask = mask.point(lambda x: 0 if x > thres else 255)
                    mask = transforms.ToTensor()(mask)
                    lrmaskrow=torch.mean(torch.cat((mask[:,0:0,:],mask[:,-1:,:]),1))
                    lrmaskcolumn=torch.mean(torch.cat((mask[:,:,0:0],mask[:,:,-1:]),2))
                    lrmean=1/2*(lrmaskcolumn+lrmaskrow)*255.0
                    
                    if((lrmean>190)):
                        imagelr_maskr=mask
                        imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                        imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                        imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                        mask= imagelr_maskr#对mask做反转
                    img_tensor = torch.cat((img[i], mask), 0)
                    img[i]=img_tensor
                
            #img_tensor = img_tensor.unsqueeze(0)
            return img
        def transformlr_(path):
            img = Image.open(path)
            #img = img.resize((256, 32), Image.BICUBIC)
            #img = img.resize((64, 16), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            '''if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                lrmaskrow=torch.mean(torch.cat((mask[:,0:0,:],mask[:,-1:,:]),1))
                lrmaskcolumn=torch.mean(torch.cat((mask[:,:,0:0],mask[:,:,-1:]),2))
                lrmean=1/2*(lrmaskcolumn+lrmaskrow)*255.0
                
                if((lrmean>190)):
                    imagelr_maskr=mask
                    imagelr_maskr[torch.where(imagelr_maskr==0)]=2
                    imagelr_maskr[torch.where(imagelr_maskr==1)]=0
                    imagelr_maskr[torch.where(imagelr_maskr==2)]=1
                    mask= imagelr_maskr#对mask做反转
                img_tensor = torch.cat((img_tensor, mask), 0)'''
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        elif self.args.rec=='autostr':
            autostr,autostr_info=self.AutoSTR_init()
            autostr.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        j=0
        
        sr_count=0
        sr_correct=0
        lr_correct=0
        test_dataset,test_loader=self.get_test_data_regular(self.args.demo_dir)
        import ipdb;ipdb.set_trace()
        for i, (images_lr, image_tensors, labels, path) in enumerate(tqdm(test_loader)):
            images_lr0 = images_lr
            batch_size=images_lr.size(0)
            images_lr = transform_(images_lr,image_tensors)
            
            images_lr0 = images_lr0.to(self.device)
            #images_lr = transforms.ToTensor()(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
            #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
            sr_count=sr_count+1
            images_sr0,images_sr,_ = model(images_lr)
            

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                '''moran_input = self.parse_moran_data(images_lr[:, :3, :, :])
                #import ipdb;ipdb.set_trace()
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                #import ipdb;ipdb.set_trace()
                pred_str_lr = sim_preds.split('$')[0]'''
                #pred_str_lr = [pred.split('$')[0] for pred in sim_preds]
                #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                    moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                        debug=True)
                    preds, preds_reverse = moran_output[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                    pred_str_sr = sim_preds.split('$')[0]
                #pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    moran_input_lr = self.parse_moran_data(images_lr0[:, :3, :, :])
                    moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3], test=True,
                                        debug=True)
                    preds_lr, preds_reverse_lr = moran_output_lr[0]
                    _, preds_lr = preds_lr.max(1)
                    sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                    #pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
                    pred_str_lr = sim_preds.split('$')[0]
            elif self.args.rec == 'aster':
                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                    aster_output_sr = aster(aster_dict_sr)
                    pred_rec_sr = aster_output_sr['output']['pred_rec']
                    pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
               

                if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                    aster_dict_lr = self.parse_aster_data(images_lr0[:, :3, :, :])
                    aster_output_lr = aster(aster_dict_lr)
                    pred_rec_lr = aster_output_lr['output']['pred_rec']
                    pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                crnn_input_lr = self.parse_crnn_data(images_lr0[:, :3, :, :])
                crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * batch_size)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            elif self.args.rec=='autostr':
                '''autostr_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                autostr_output_sr = aster(autostr_dict_sr)
                pred_rec_sr = autostr_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, autostr_dict_sr['rec_targets'], dataset=autostr_info)
               '''

                autostr_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                autostr_output_lr = autostr(autostr_dict_lr)
                pred_rec_lr = autostr_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, autostr_dict_lr['rec_targets'], dataset=autostr_info)
            j=j+1
            #import ipdb;ipdb.set_trace()
            
            #self.test_display_IC15(images_lr[:,:3,:,:],images_sr[:,:3,:,:],pred_str_lr,pred_str_sr,label,str_filt)
            #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
            '''if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                if(len(pred_str_lr)>0):
                    if(str_filt(pred_str_lr[0], 'lower') == str_filt(label, 'lower')):
                            lr_correct=lr_correct+1
            if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
            #if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                if(len(pred_str_sr)>0):
                    if(str_filt(pred_str_sr[0], 'lower') == str_filt(label, 'lower')):
                            sr_correct=sr_correct+1'''
            '''if(len(pred_str_lr)>0):
                if(str_filt(pred_str_lr[0], 'lower') == str_filt(label, 'lower')):
                    n_correct=n_correct+1'''
            '''if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                if(str_filt(pred_str_lr, 'lower') == str_filt(label, 'lower')):
                        lr_correct=lr_correct+1
            if((images_lr.size(2)<16)|(images_lr.size(3)<64)):
                if(str_filt(pred_str_sr, 'lower') == str_filt(label, 'lower')):
                        sr_correct=sr_correct+1'''
            for pred, gt in zip(pred_str_sr, labels):
            #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                if(str_filt(pred, 'lower') == str_filt(gt, 'lower')):
                        lr_correct=lr_correct+1
            for pred, gt in zip(pred_str_sr, labels):
            #if(((images_lr.size(2)*images_lr.size(3))<(64*16))):
                if(str_filt(pred, 'lower') == str_filt(gt, 'lower')):
                        sr_correct=sr_correct+1
            #print(pred_str_sr[0], '===>', label)
            torch.cuda.empty_cache()
        sum_images = len(test_loader)
        accuracy_sr=round(sr_correct / sum_images, 4)
        accuracy_lr=round(lr_correct / sum_images, 4)
        #accuracy=round(n_correct / sum_images, 4)
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        print('fps=', fps)
        #print('sr_count=', sr_count)
        print('accuracy_lr=', accuracy_lr)
        print('accuracy_sr=', accuracy_sr)
        #print('accuracy=',accuracy)
        

if __name__ == '__main__':
    embed()