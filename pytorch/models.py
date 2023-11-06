#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchlibrosa.stft import Spectrogram, LogmelFilterBank # use 'magphase' to extract phase informatinon with magnitude
# ref: https://github.com/qiuqiangkong/torchlibrosa
from pytorch_utils import move_data_to_device
from transformer_encoder import TransformerEncoder
from transformer_encoder.utils import PositionalEncoding



# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[3]:


# initialize and conv block
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)
        
def init_ln(module):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
"""
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        
        x = x.transpose(0,1) # [batch_size, seq_len, embedding_dim] → [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0,1)  # [seq_len, batch_size, embedding_dim] → [batch_size, seq_len, embedding_dim]
        return self.dropout(x)
"""

"""
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=8):
        super(TransformerEncoder, self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead) #,batch_first=True) torch==1.8.0ではbatch_firstが使えない、、
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        
    def forward(self, inputs):
        inputs = inputs.transpose(0,1)
        output = self.transformer_encoder(inputs)
        output = output.transpose(0,1)
        return output
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)
        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        #print(x)
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        
        return x


# In[4]:


# アコースティックモデル
class AcousticModelCRnn8Dropout(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticModelCRnn8Dropout, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)
        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)
        
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output 
    
# アコースティックモデル(Transformer バージョン)

# アコースティックモデル
class AcousticModelTransformer(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticModelTransformer, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        
        self.ln1 = nn.LayerNorm(768)
        self.pe = PositionalEncoding(d_model=768, dropout=0.1, max_len=2000)
        
        self.encoder_layer = TransformerEncoder(d_model=768, d_ff=1024, n_heads=8, n_layers=6, dropout=0.1)

        self.fc1 = nn.Linear(768, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, classes_num, bias=False)
        
        
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_ln(self.ln1)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)
        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) # [batch_size, 128, 1001, 14]
        
        x = x.transpose(1, 2).flatten(2)  # [batch_size, 1001, 1792]

        x = F.relu(self.fc5(x)) # [batch_size, 1001, 768]
        x = self.ln1(x)
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)  # [batch_size, 1001, 768]

        x = self.pe(x) # [batch_size, 1001, 768]
        
        x = self.encoder_layer(x, mask=None)  # [batch_size, 1001, 768]
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        x = self.fc1(x) # [batch_size, 1001, 512]
        x = self.fc2(x) # [batch_size, 1001, 256]
        
        output = torch.sigmoid(self.fc3(x)) # [batch_size, 256, 88]
        
        return output

# In[11]:


# regression
class Regress_onset_offset_frame_velocity_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Regress_onset_offset_frame_velocity_CRNN, self).__init__()
        
        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        midfeat = 1792
        momentum = 0.01
        
        # Spectrogram
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
            hop_length=hop_size, win_length=window_size, window=window,
            center=center, pad_mode=pad_mode, freeze_parameters=True)
        
        # Logmel
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref,
            amin=amin, top_db=top_db, freeze_parameters=True)
        
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        
        self.frame_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        #self.velocity_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.velocity_model = AcousticModelTransformer(classes_num, midfeat, momentum)
        
        # after CRNN block
        # only onset and frame is required
        # "attention": figs (high_resolution and exploring transformer's pottential) is different but same model expect velocity network
        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1,
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)
        
        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1,
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()
        
    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)
        
    def forward(self, input):
        """
        Args:
            input: (batch_size, data_length)
        
        Outputs:
            output_dict: dict, {
              'reg_onset_output': (batch_size, time_steps, classes_num),
              'reg_offset_output': (batch_size, time_steps, classes_num),
              'frame_output': (batch_size, time_steps, classes_num),
              'velocity_output': (batch_size, time_steps, classes_num)
            }
        """
        
        x = self.spectrogram_extractor(input) # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x) # (batch_size, 1, time_step, freq_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        frame_output = self.frame_model(x)
        reg_onset_output = self.reg_onset_model(x)
        reg_offset_output = self.reg_offset_model(x)
        velocity_output = self.velocity_model(x)
        
        # Concatenete veloacity and onset output to regress final onset output
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""
        
        # concatenate on/offset and frame outputs to classifier final pitch output
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x)) # (batch_size,  time_steps, classes_num)
        
        output_dict = {
            'reg_onset_output': reg_onset_output,
            'reg_offset_output': reg_offset_output,
            'frame_output': frame_output,
            'velocity_output': velocity_output}
        
        return output_dict


# In[18]:


# regression
class Regress_onset_offset_frame_velocity_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Regress_onset_offset_frame_velocity_CRNN, self).__init__()
        
        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        midfeat = 1792
        momentum = 0.01
        
        # Spectrogram
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
            hop_length=hop_size, win_length=window_size, window=window,
            center=center, pad_mode=pad_mode, freeze_parameters=True)
        
        # Logmel
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref,
            amin=amin, top_db=top_db, freeze_parameters=True)
        
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        
        self.frame_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        #self.velocity_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.velocity_model = AcousticModelTransformer(classes_num, midfeat, momentum)
        
        # after CRNN block
        # only onset and frame is required
        # "attention": figs (high_resolution and exploring transformer's pottential) is different but same model expect velocity network
        self.reg_onset_gru = nn.GRU(input_size=88 * 2+1, hidden_size=256, num_layers=1,
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)
        
        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1,
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()
        
    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)
        

        
    def forward(self, input):
        """
        Args:
            input: (batch_size, data_length)
        
        Outputs:
            output_dict: dict, {
              'reg_onset_output': (batch_size, time_steps, classes_num),
              'reg_offset_output': (batch_size, time_steps, classes_num),
              'frame_output': (batch_size, time_steps, classes_num),
              'velocity_output': (batch_size, time_steps, classes_num)
            }
        """
        
        # phase特徴量を求める
        # 位相スペクトログラムを作成
        (pt_stft_real, pt_stft_imag) = self.pt_stft_extractor.forward(input)
        phase = torch.atan2(pt_stft_imag, pt_stft_real) # (1,1,1001(時間),1025(周波数))  ***atan2の引数は第一：y軸(imaginary part) 第二：x軸(real part)***
        # 横方向で微分 (1回目)
        IF = self.calc_div(phase)
        IF = self.phase_wrapping_torch(IF)

        # 横方向で微分(2回目)
        IF = self.calc_div(IF, change=False)
        IF = self.phase_wrapping_torch(IF)
        IF = torch.abs(IF)
        IF = torch.sum(IF, axis=2).transpose(1,2) # [batch_size, time_steps, n_dim(1)]
        
        
        x = self.spectrogram_extractor(input) # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x) # (batch_size, 1, time_step, freq_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        frame_output = self.frame_model(x)
        reg_onset_output = self.reg_onset_model(x)
        reg_offset_output = self.reg_offset_model(x)
        velocity_output = self.velocity_model(x)
        
        # Concatenete veloacity and onset output to regress final onset output
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach(), IF), dim=2)
        (x, _) = self.reg_onset_gru(x)
        print(x.shape)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""
        
        # concatenate on/offset and frame outputs to classifier final pitch output
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x)) # (batch_size,  time_steps, classes_num)
        
        output_dict = {
            'reg_onset_output': reg_onset_output,
            'reg_offset_output': reg_offset_output,
            'frame_output': frame_output,
            'velocity_output': velocity_output}
        
        return output_dict


# In[1]:


# This model is not trained, but is combined from the trained note and pedal models.
class Note_pedal(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        """The combination of note and pedal model.
        """
        super(Note_pedal, self).__init__()

        self.note_model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second, classes_num)
        self.pedal_model = Regress_pedal_CRNN(frames_per_second, classes_num)

    def load_state_dict(self, m, strict=False):
        self.note_model.load_state_dict(m['note_model'], strict=strict)
        self.pedal_model.load_state_dict(m['pedal_model'], strict=strict)

    def forward(self, input):
        note_output_dict = self.note_model(input)
        pedal_output_dict = self.pedal_model(input)

        full_output_dict = {}
        full_output_dict.update(note_output_dict)
        full_output_dict.update(pedal_output_dict)
        return full_output_dict

