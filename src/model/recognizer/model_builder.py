from __future__ import absolute_import

from PIL import Image
import numpy as np
from collections import OrderedDict
import sys
import time
from .resnet_aster import *
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

#from . import create
from .attention_recognition_head import AttentionRecognitionHead
from .sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .tps_spatial_transformer import TPSSpatialTransformer1

#from config import get_args
#global_args = get_args(sys.argv[1:])
tps_inputsize = [32, 64]
tps_outputsize = [32, 100]
num_control_points = 20
tps_margins = [0.05, 0.05]
beam_width = 5

class ModelBuilder(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels, eos, args, STN_ON=False):
    super(ModelBuilder, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels
    self.eos = eos
    self.STN_ON = STN_ON
    self.tps_inputsize = tps_inputsize
    self.encoder = ResNet_ASTER(self.arch)

    '''self.encoder = create(self.arch,
                      with_lstm=True,
                      n_group=1,
                      network_id=args.network_id,
                      path_configs_file=args.path_configs_file,
                      config_file=args.nas_config_file)'''
                      
    encoder_out_planes = self.encoder.out_planes

    self.decoder = AttentionRecognitionHead(
                      num_classes=rec_num_classes,
                      in_planes=encoder_out_planes,
                      sDim=sDim,
                      attDim=attDim,
                      max_len_labels=max_len_labels)
    self.rec_crit = SequenceCrossEntropyLoss()
    if self.STN_ON:
            self.tps = TPSSpatialTransformer1(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))
            self.stn_head = STNHead(
                in_planes=3,
                num_ctrlpoints=num_control_points,
                activation='none')

    '''if self.STN_ON:
      self.tps = TPSSpatialTransformer(
        output_image_size=tuple(global_args.tps_outputsize),
        num_control_points=global_args.num_control_points,
        margins=tuple(global_args.tps_margins))
      self.stn_head = STNHead(
        in_planes=3,
        num_ctrlpoints=global_args.num_control_points,
        activation=global_args.stn_activation)'''

  def test_single_pic(self, x):
    t0 = time.time()
    if self.STN_ON:
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
    tps_cost = time.time() - t0
  
    t0 = time.time()
    encoder_feats = self.encoder(x)
    cnn_cost = time.time() - t0

    encoder_feats = encoder_feats.contiguous()

    t0 = time.time()
    greedy_rec_pred = self.decoder.greedy_sample(encoder_feats)
    rnn_cost = time.time() - t0
    return greedy_rec_pred, tps_cost, cnn_cost, rnn_cost

  def test_pic_with_beamsearch(self, x):
    t0 = time.time()
    if self.STN_ON:
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_image_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
    tps_cost = time.time() - t0

    t0 = time.time()
    encoder_feats = self.encoder(x)
    cnn_cost = time.time() - t0

    t0 = time.time()
    rec_preds, rec_pred_scores = self.decoder.beam_search(encoder_feats, global_args.beam_width, self.eos)
    rnn_cost = time.time() - t0
    return rec_preds, tps_cost, cnn_cost, rnn_cost

  def forward(self, input_dict):
    return_dict = {}
    return_dict['losses'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']

    # rectification
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
      if not self.training:
        # save for visualization
        return_dict['output']['ctrl_points'] = ctrl_points
        return_dict['output']['rectified_images'] = x

    encoder_feats = self.encoder(x)
    encoder_feats = encoder_feats.contiguous()

    if self.training:
      rec_pred = self.decoder([encoder_feats, rec_targets, rec_lengths])
      loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
      return_dict['losses']['loss_rec'] = loss_rec
    else:
      rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, 5, self.eos)
      rec_pred_ = self.decoder([encoder_feats, rec_targets, rec_lengths])
      loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)
      return_dict['losses']['loss_rec'] = loss_rec
      return_dict['output']['pred_rec'] = rec_pred
      return_dict['output']['pred_rec_score'] = rec_pred_scores

    # pytorch0.4 bug on gathering scalar(0-dim) tensors
    for k, v in return_dict['losses'].items():
      return_dict['losses'][k] = v.unsqueeze(0)
    return return_dict

  def test(self, x):
    '''Test single picture'''
    pass