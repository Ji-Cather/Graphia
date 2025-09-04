import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .embed import DataEmbedding



class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, input_time_len, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'),
                T = 'm'
                ):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.attn = attn
        self.output_attention = output_attention
        self.input_time_len = input_time_len

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, T, freq, input_time_len, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, T, freq, input_time_len, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.skip_connection = nn.Linear(1 + self.seq_len, c_out)
        self.init_weights()
        
    def init_weights(self):
        """初始化skip_connection的权重和偏置"""
        with torch.no_grad():
            # 设置权重为[1/self.input_len]
            self.skip_connection.weight.fill_(1.0/self.seq_len)
            self.skip_connection.weight[:, 0].fill_(0.0)
            # 设置偏置为全0
            self.skip_connection.bias.fill_(0.0)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
       
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerTranverse(Informer):
    #input 1, bwr, time * feature
    def __init__(self, 
                 bwr, 
                 node_feature_dim, 
                 input_len, 
                 src_node_out_feature_dim:int = 1, # 1 for degree, multi for other features
                factor=5, 
                d_model=512, 
                n_heads=8, 
                e_layers=3, 
                d_layers=2, 
                d_ff=512, 
                dropout=0.0, 
                attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        # 输入shape: [B, S, D] (B, seq_len, enc_in) -> [1, B, S*D]
        # 输出                                         [1, B, c_out*out_len]    -> [B, out_len, c_out]
        
        # 输入shape: [B, bwr, SD] -> [B, bwr, src_out_dim]
        self.pred_len = 1
        self.input_len = input_len
        self.src_node_out_feature_dim = src_node_out_feature_dim
       
        super(InformerTranverse, self).__init__(node_feature_dim, 
                                                node_feature_dim, 
                                                input_len,
                                                src_node_out_feature_dim, 
                                                bwr, 
                                                bwr, 
                                                bwr,  
                                                factor, 
                                                d_model, n_heads, e_layers, d_layers, d_ff, dropout, attn, embed, freq, activation, output_attention, distil, mix, device)
        
        self.skip_connection = nn.Linear(self.input_len*self.src_node_out_feature_dim, 
                                         src_node_out_feature_dim)
        self.init_weights()
        
    def init_weights(self):
        """初始化skip_connection的权重和偏置"""
        with torch.no_grad(): # weight ?
            # 设置权重为[1/self.input_len]
            self.skip_connection.weight.fill_(1e-3)
            for feature_id in range(self.src_node_out_feature_dim):
                start_idx = feature_id * self.input_len + 1
                end_idx = start_idx + self.input_len
                self.skip_connection.weight[feature_id, start_idx:end_idx].fill_(1.0/(self.input_len))
            self.skip_connection.weight[:, 0].fill_(1e-3)
            # 设置偏置为全0
            self.skip_connection.bias.fill_(1e-3)
            
        
    def forward(self, 
                x_enc, 
                x_mark_enc, 
                x_dec, 
                x_mark_dec, 
                enc_self_mask=None, 
                dec_self_mask=None, 
                dec_enc_mask=None):
        # 重新排列维度，使得相同位置的元素相邻
        # 对所有输入进行相同的变换操作：先调整维度顺序，然后重塑并增加一个维度
        x_enc = x_enc.permute(0, 2, 1).reshape(x_enc.shape[0], -1).unsqueeze(0)
        x_mark_enc = x_mark_enc.permute(0, 2, 1).reshape(x_mark_enc.shape[0], -1).unsqueeze(0)
        x_dec = x_dec.permute(0, 2, 1).reshape(x_dec.shape[0], -1).unsqueeze(0)
        x_mark_dec = x_mark_dec.permute(0, 2, 1).reshape(x_mark_dec.shape[0], -1).unsqueeze(0)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        bwr = dec_out.shape[1]
        batch_size = dec_out.shape[0]
        # 添加 skip connection
        dec_out_skip = torch.cat((dec_out, x_dec), dim=-1)[:, :, 
                                :1+ self.input_len*self.src_node_out_feature_dim]
        dec_out_skip = dec_out_skip.squeeze(0)
        residual = self.skip_connection(dec_out_skip).reshape(batch_size,bwr,-1)
      
        output = residual
        if self.output_attention:
            return output, attns
        else:
            return output # [B, L, D]
        
    def forward_encoder(self, 
                x_enc, 
                x_mark_enc_input, 
                x_mark_dec_one,
                enc_self_mask=None, 
                dec_self_mask=None, 
                dec_enc_mask=None):
        # 重新排列维度，使得相同位置的元素相邻
        # 对所有输入进行相同的变换操作：先调整维度顺序，然后重塑并增加一个维度
        # 输入shape: [B, bwr, SD] -> [B, bwr, src_out_dim]
        x_enc = x_enc
        x_mark_enc = x_mark_enc_input
        x_dec = x_enc
        x_mark_dec = torch.concat((x_mark_enc_input[:,:,1:], x_mark_dec_one.unsqueeze(-1)), dim=-1)
        x_mark_dec = x_mark_dec
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # B, bwr//2, d_model

        dec_out = self.dec_embedding(x_dec, x_mark_dec) # B, bwr, d_model
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        bwr = dec_out.shape[1]
        batch_size = dec_out.shape[0]
        # 添加 skip connection
        dec_out_skip = torch.cat((dec_out, x_dec), dim=-1)[:, :, 
                                :1+ self.input_len*self.src_node_out_feature_dim]
        dec_out_skip = dec_out_skip.reshape(batch_size* bwr, -1)
        output = self.skip_connection(dec_out_skip).reshape(batch_size,bwr,-1)
        if self.output_attention:
            return output, attns
        else:
            return output # [B, L, D]
        
        
    def cacu_loss(self, pred, target, loss_type='l1'):
        if loss_type == 'l1':
            loss = F.l1_loss(pred, target)
        elif loss_type == 'l2':
            loss = F.mse_loss(pred, target)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(pred, target)
        else:
            raise NotImplementedError()
        return loss
    
    
class InformerDecoder(nn.Module):
    #input 1, bwr, time * feature
    def __init__(self, 
                 bwr, 
                 node_feature_dim, 
                 input_len, 
                 pred_len,
                 src_node_out_feature_dim:int = 1, # 1 for degree, multi for other features
                factor=5, 
                d_model=512, 
                n_heads=8, 
                e_layers=3, 
                d_layers=2, 
                d_ff=512, 
                dropout=0.0, 
                attn='prob', 
                embed='fixed', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'),
                T = 'm',
                freq='d',
                max_len=10000
                ):
        # 输入shape: [B, S, D] (B, seq_len, enc_in) -> [1, B, S*D]
        # 输出                                         [1, B, c_out*out_len]    -> [B, out_len, c_out]
        
        # 输入shape: [B, bwr, SD] -> [B, bwr, src_out_dim]
        self.output_attention = output_attention
        self.pred_len = pred_len
        self.input_len = input_len
        self.src_node_out_feature_dim = src_node_out_feature_dim
        super(InformerDecoder, self).__init__()

        self.enc_embedding = DataEmbedding(node_feature_dim, d_model, embed, T, freq, input_len, dropout, max_len)
        self.dec_embedding = DataEmbedding(node_feature_dim, d_model, embed, T, freq, input_len, dropout, max_len)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, 
                                    src_node_out_feature_dim, 
                                    bias=True) 
        
        self.skip_connection = nn.Linear((self.input_len+1)*self.src_node_out_feature_dim, 
                                         self.pred_len*src_node_out_feature_dim)
        self.init_weights()
        
    def init_weights(self):
        """初始化skip_connection的权重和偏置"""
        with torch.no_grad(): # weight ?
            self.skip_connection.weight.fill_(1e-3)
            
            for feature_id in range(self.src_node_out_feature_dim):
                start_idx = feature_id * self.input_len + self.src_node_out_feature_dim
                end_idx = start_idx + self.input_len
                self.skip_connection.weight[:, start_idx:end_idx].fill_(1.0/(self.input_len))
                
            # 将权重的每一列复制到所有行
            for feature_id in range(self.src_node_out_feature_dim):
                self.skip_connection.weight[:, feature_id].fill_(1e-3)
            # 设置偏置为全0
            self.skip_connection.bias.fill_(1e-3)
            
    def forward_encoder(self, 
                x_enc, 
                x_mark_enc, 
                x_mark_dec,
                enc_self_mask=None, 
                dec_self_mask=None, 
                dec_enc_mask=None):
        # 重新排列维度，使得相同位置的元素相邻
        # 对所有输入进行相同的变换操作：先调整维度顺序，然后重塑并增加一个维度
        # 输入shape: [B, bwr, SD] -> [B, bwr, src_out_dim]
        x_enc = x_enc
        x_mark_enc = x_mark_enc
        x_dec = x_enc
        x_mark_dec = x_mark_enc
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # B, bwr//2, d_model

        dec_out = self.dec_embedding(x_dec, x_mark_dec) # B, bwr, d_model
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        bwr = dec_out.shape[1]
        batch_size = dec_out.shape[0]
        # 添加 skip connection
        dec_out_skip = torch.cat((dec_out, x_dec), dim=-1)[:, :, 
                                :(self.input_len+1)*self.src_node_out_feature_dim]
        dec_out_skip = dec_out_skip.reshape(batch_size* bwr, -1)
        output = self.skip_connection(dec_out_skip).reshape(batch_size, bwr,-1)
        if self.output_attention:
            return output, attns
        else:
            return output # [B, L, D]
        
        
    def cacu_loss(self, pred, target, loss_type='l1'):
        if loss_type == 'l1':
            loss = F.l1_loss(pred, target)
        elif loss_type == 'l2':
            loss = F.mse_loss(pred, target)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(pred, target)
        else:
            raise NotImplementedError()
        return loss
        
        

        
        
        