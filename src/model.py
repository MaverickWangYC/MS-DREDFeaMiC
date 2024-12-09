import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from einops import rearrange
from tqdm import tqdm
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import numpy as np

USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0
batch_size = 16
last_batch_size = 16 # only for the very last batch of the dataset
different_batch_size = False
h_new = None
temp_buffer = None
#d_model = 8
#state_size = 128 # Example state size
seq_len = 1
class Attention(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., pred=True):
        super().__init__()
        self.in_features = in_features
        #out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(0.01)
        self.pred = pred
        self.LayerNorm1 = nn.LayerNorm(in_features, 1e-6)
        if pred==True:
            self.fc2 = nn.Linear(hidden_features,1)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)

        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x0 = x

        qq = self.q(x).reshape(-1,self.in_features)


        q = qq.unsqueeze(2)

        kk = self.k(x).reshape(-1,self.in_features)
        k = kk.unsqueeze(2)
        vv = self.v(x).reshape(-1, self.in_features)
        v = vv.unsqueeze(2)
        attn = (q @ k.transpose(-2, -1))
        #print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        #print(x.size())

        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.drop(x)
        if self.pred==False:
            x += x1
            #x = self.LayerNorm1(x)
        x = x.squeeze(0)

        return x

class baselinemlp(nn.Module):
    def __init__(self, in_features=13501, median_size=512, output_size = 3):
        super().__init__()
        """
        #out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,in_features)
        self.act = act_layer(0.01)
        self.fc2 = nn.Linear(in_features,hidden_features)

        self.pred = pred
        #self.LayerNorm1 = nn.LayerNorm(in_features, 1e-6)
        if pred==True:
            self.fc3 = nn.Linear(hidden_features,1)
        else:
            self.fc3 = nn.Linear(hidden_features, in_features)

        self.drop = nn.Dropout(drop)
        """
        self.median_size = median_size
        self.output_size = output_size
        self.Linear_0 = nn.Linear(in_features,self.median_size)
        #self.fc2 = nn.Linear(in_features,in_features)
        self.Linear_1 = nn.Linear(self.median_size,self.median_size)
        self.Linear_2 = nn.Linear(self.median_size, self.median_size // 2)
        self.Linear_3 = nn.Linear(self.median_size // 2, self.median_size // 4)
        #self.fc4 = nn.Linear(in_features, in_features)
        self.output = nn.Linear(self.median_size // 4,self.output_size)
        #self.LeakyReLU = nn.LeakyReLU(0.01)
        self.LeakyReLU = nn.LeakyReLU(0.01)


    def forward(self, x):

        x = self.Linear_0(x)
        x = self.LeakyReLU(x)

        x = self.Linear_1(x)
        x = self.LeakyReLU(x)

        x = self.Linear_2(x)
        x = self.LeakyReLU(x)

        x = self.Linear_3(x)
        x = self.LeakyReLU(x)

        x = self.output(x)
        x = self.LeakyReLU(x)
        x = x.reshape(-1,self.output_size)
        return x, x, x, x, x

class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.Linear_0 = nn.Linear(input_size,hidden_size)
        self.rnn_1 = nn.RNN(hidden_size, hidden_size, batch_first=False)
        self.rnn_2 = nn.RNN(hidden_size, hidden_size // 2, batch_first=False)
        self.rnn_3 = nn.RNN(hidden_size // 2, hidden_size // 4, batch_first=False)
        self.fc1 = nn.Linear(hidden_size // 4,output_size)
        self.act = nn.LeakyReLU(0.01)


    def forward(self,x):
        x = self.Linear_0(x)
        x = self.act(x)
        out,_ = self.rnn_1(x)
        out,_ = self.rnn_2(out)
        out,_ = self.rnn_3(out)
        out = self.fc1(out)
        out = self.act(out)
        #out = self.act1(out)
        return out, out, out, out, out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1,num_layers=2):
        super(LSTM, self).__init__()
        self.Linear_0 = nn.Linear(input_size,hidden_size)
        self.lstm_1 = nn.LSTM(hidden_size, hidden_size,num_layers)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size // 2, num_layers)
        self.lstm_3 = nn.LSTM(hidden_size // 2, hidden_size // 4, num_layers)
        self.reg = nn.Linear(hidden_size // 4, output_size)
        self.act = nn.LeakyReLU(0.01)
    def forward(self,x):
        x = self.Linear_0(x)
        x = self.act(x)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        x, _ = self.lstm_3(x)

        x = self.reg(x)
        x = self.act(x)
        return x, x ,x, x, x

class S6(nn.Module):
    def __init__(self,seq_len, d_model, state_size):
        super(S6, self).__init__()
        #self.batch_size = batch_size
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, state_size)
        self.fc3 = nn.Linear(d_model, state_size)
        self.fc4 = nn.Linear(d_model, state_size)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)


        # h [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model)
        self.relu = nn.ReLU()
    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))

        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2 MAMBA paper
        #x = x.reshape(self.batch_size,-1,1)#16 4 1
        self.B = self.fc2(x)#16 4 128
        self.C = self.fc3(x)#16 4 128
        self.delta = F.softplus(self.fc1(x))


        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:

            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True

                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,
                                                                                                               "b l d -> b l d 1") * self.dB


            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y [batch_size, seq_len, d_model]
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)


            #self.y = x + self.y
            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
            #print(self.y)
            return self.y


        else:
            # h [batch_size, seq_len, d_model, state_size]
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y [batch_size, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str = 'cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, median_size):
        super(MambaBlock, self).__init__()

        self.inp_proj = nn.Linear(d_model, median_size)
        self.out_proj = nn.Linear(median_size, median_size)
        # For residual skip connection
        self.D = nn.Linear(d_model, median_size)

        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True

        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, median_size, state_size)

        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1)

        # Add linear layer for conv output
        self.conv_linear = nn.Linear(median_size, median_size)
        self.Linear_0 = nn.Linear(median_size,median_size)
        # rmsnorm
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
        """
        # Refer to Figure 3 in the MAMBA paper

        x = self.norm(x)

        x_proj = self.inp_proj(x)

        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)
        # x_conv = self.Linear_0(x_proj)
        x_conv_act = F.silu(x_conv)

        # Add linear layer for conv output

        x_conv_out = self.conv_linear(x_conv_act)


        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))

        x_combined = x_act * x_residual

        x_out = self.out_proj(x_combined)

        return x_out

class Transformer(nn.Module):
    def __init__(self, seq_len, d_model, state_size, median_size, output_size):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.output_size = output_size
        self.median_size = median_size
        self.qkv_size = 8

        # self.Linear_0 = nn.Linear(self.d_model, self.median_size)
        self.Linear_1 = nn.Linear(self.d_model, self.median_size)
        self.Linear_2 = nn.Linear(self.median_size, self.median_size)
        self.Linear_3 = nn.Linear(self.median_size, self.median_size)
        self.Linear_4 = nn.Linear(self.median_size, self.median_size)
        self.Linear_5 = nn.Linear(self.median_size, self.median_size)

        self.output = nn.Linear(self.median_size, self.output_size)

        # self.transformer_1 = SelfAttention(self.median_size, heads=self.qkv_size)
        # self.transformer_2 = SelfAttention(self.median_size, heads=self.qkv_size)

        self.transformer_1 = Attention(in_features=self.median_size, hidden_features=self.median_size, act_layer=nn.LeakyReLU, drop=0,
                                      pred=False)
        self.transformer_2 = Attention(in_features=self.median_size, hidden_features=self.median_size, act_layer=nn.LeakyReLU, drop=0,
                                      pred=False)
        self.transformer_3 = Attention(in_features=self.median_size, hidden_features=self.median_size, act_layer=nn.LeakyReLU, drop=0,
                                      pred=False)


        self.layer_norm_1 = nn.LayerNorm(self.median_size)
        self.layer_norm_2 = nn.LayerNorm(self.median_size)
        self.layer_norm_3 = nn.LayerNorm(self.median_size)
        self.layer_norm_4 = nn.LayerNorm(self.median_size)

        self.norm_1 = RMSNorm(self.median_size)
        self.norm_2 = RMSNorm(self.median_size)
        self.norm_3 = RMSNorm(self.median_size)
        self.norm_4 = RMSNorm(self.median_size)
        # self.norm_1 = RMSNorm(self.median_size)
        # self.norm_2 = RMSNorm(self.median_size)
        self.leaky = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.0)
        self.gelu = nn.GELU()
    def forward(self,x):
        x = self.Linear_1(x)
        x1 = self.leaky(x)

        x = self.transformer_1(x)
        x = self.layer_norm_1(x)
        x2 = x + x1
        x = self.transformer_2(x2)
        x = self.layer_norm_2(x)
        x3 = x + x2
        x = self.transformer_3(x3)
        x = self.layer_norm_3(x)
        x = x + x3

        x = self.output(x)
        x = self.leaky(x)
        x = x.reshape(-1, self.output_size)
        return x, x, x, x, x

class MSDREDFeaMiC(nn.Module):
    def __init__(self, seq_len, d_model, state_size, median_size, output_size):
        super(MSDREDFeaMiC, self).__init__()
        self.d_model = d_model
        self.median_size = median_size
        self.output_size = output_size
        # self.mamba_block0 = MambaBlock(seq_len, self.d_model, state_size, self.median_size)
        self.mamba_block1 = MambaBlock(seq_len, self.median_size, state_size, self.median_size)
        self.mamba_block2 = MambaBlock(seq_len, self.median_size, state_size, self.median_size // 2)
        self.mamba_block3 = MambaBlock(seq_len, self.median_size // 2, state_size, self.median_size // 4)

        self.Linear_0 = nn.Linear(self.d_model, self.median_size)

        self.Linear_1 = nn.Linear(self.median_size, self.median_size)
        self.Linear_2 = nn.Linear(self.median_size, self.median_size // 2)
        self.Linear_3 = nn.Linear(self.median_size // 2, self.median_size // 4)
        self.Linear_4 = nn.Linear(self.median_size // 4, self.median_size // 2)
        self.Linear_5 = nn.Linear(self.median_size // 2, self.median_size)

        self.Linear_recon = nn.Linear(self.median_size, self.d_model)

        self.bn_1 = nn.BatchNorm1d(self.d_model)
        self.bn_2 = nn.BatchNorm1d(self.median_size)

        self.output_1 = nn.Linear(self.median_size, output_size)
        self.output_2 = nn.Linear(self.median_size // 2, output_size)
        self.output_3 = nn.Linear(self.median_size // 4, output_size)
        self.output_sigmoid = nn.Linear(self.median_size // 4, 1)

        self.dim_1 = nn.Linear(self.median_size * 2, self.median_size)
        self.dim_2 = nn.Linear(self.median_size, self.median_size // 2)
        self.dim_3 = nn.Linear(self.median_size // 2, self.median_size // 4)

        self.relu = nn.ReLU()
        self.act = nn.LeakyReLU(0.01)
        self.norm_1 = RMSNorm(self.median_size)
        self.norm_2 = RMSNorm(self.median_size // 2)
        self.norm_3 = RMSNorm(self.median_size // 4)

        self.bnNorm_1 = nn.LayerNorm(self.median_size)
        self.bnNorm_2 = nn.LayerNorm(self.median_size // 2)
        self.bnNorm_3 = nn.LayerNorm(self.median_size // 4)
        self.bnNorm_4 = nn.LayerNorm(self.median_size // 2)
        self.bnNorm_5 = nn.LayerNorm(self.median_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.bn_1(x)
        x = self.Linear_0(x)
        x = self.act(x)
        x_encoder = x
        x0_r = x.reshape(-1, 1, self.median_size)

        x = self.Linear_1(x0_r)
        x1 = self.act(x)
        x1 = self.bnNorm_1(x1)

        x1_r = x1.reshape(-1, 1, self.median_size)

        x = self.Linear_2(x1_r)
        x2 = self.act(x)
        x2 = self.bnNorm_2(x2)

        x2_r = x2.reshape(-1, 1, self.median_size // 2)

        x = self.Linear_3(x2_r)
        x3 = self.act(x)
        x3 = self.bnNorm_3(x3)

        x3_r = x3.reshape(-1, 1, self.median_size // 4)

        x = self.Linear_4(x3)
        x = self.act(x)
        x = self.bnNorm_4(x)

        x = x+x2

        x = self.Linear_5(x)
        x = self.act(x)
        x = self.bnNorm_5(x)

        x = x + x1

        x_recon = self.Linear_recon(x)
        x_recon = self.act(x_recon)
        x_recon = x_recon.reshape(-1, self.d_model)

        x = x.reshape(-1,self.median_size)

        x = self.bn_2(x)

        x_decoder = x

        x = x.reshape(-1,1,self.median_size)

        x = self.mamba_block1(x)
        x = self.norm_1(x)
        x = x + x1_r


        x = self.mamba_block2(x)
        x = self.norm_2(x)
        x = x + x2_r


        x = self.mamba_block3(x)
        x = self.norm_3(x)
        x = x * x3_r


        x_triplet = x

        # 原版本
        x = self.output_3(x)
        x = self.act(x)
        x = x.reshape(-1, self.output_size)
        # x = self.output_sigmoid(x)
        # x = self.sigmoid(x)
        # x = x.reshape(-1)

        # x = self.output_sigmoid(x)
        # x = x.squeeze(dim=-1)
        # x = x.reshape(-1)
        # x = torch.softmax(x, dim = 0)

        return x, x_recon, x_triplet.reshape(-1, self.median_size//4), x_encoder, x_decoder

class pureMamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, median_size, output_size):
        super(pureMamba, self).__init__()
        self.d_model = d_model
        self.median_size = median_size
        self.output_size = output_size
        # self.mamba_block0 = MambaBlock(seq_len, self.d_model, state_size, self.median_size)
        self.mamba_block1 = MambaBlock(seq_len, self.median_size, state_size, self.median_size)
        self.mamba_block2 = MambaBlock(seq_len, self.median_size, state_size, self.median_size)
        self.mamba_block3 = MambaBlock(seq_len, self.median_size, state_size, self.median_size)

        self.Linear_1 = nn.Linear(self.d_model, self.median_size)

        self.norm_1 = RMSNorm(self.median_size)
        self.norm_2 = RMSNorm(self.median_size)
        self.norm_3 = RMSNorm(self.median_size)

        self.output_1 = nn.Linear(self.median_size, output_size)
        self.output_2 = nn.Linear(self.median_size//2, output_size)
        self.output_3 = nn.Linear(self.median_size//4, output_size)

        self.act = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = x.reshape(-1,1,self.d_model)
        x = self.Linear_1(x)
        x1 = self.act(x)
        # x = self.mamba_block0(x)

        x = self.mamba_block1(x1)
        x2 = x + x1

        x = self.mamba_block2(x2)
        x3 = x + x2

        x = self.mamba_block3(x3)
        x = x + x3
        # x = self.mamba_block3(x)

        """
        x = self.IS_Mamba_1(x0_r)
        x = self.IS_Mamba_2(x)
        x = self.IS_Mamba_3(x)
        """
        x = self.output_1(x)
        x = self.act(x)
        x = x.reshape(-1, self.output_size)

        return x, x ,x, x, x