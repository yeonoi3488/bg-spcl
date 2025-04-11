import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from models.layers import Conv2dWithConstraint


class channel_attention(nn.Module):
    def __init__(self, temporal_size, num_channels, inter=30):
        super(channel_attention, self).__init__()
        self.temporal_size = temporal_size
        self.inter = inter
        self.extract_sequence = int(self.temporal_size / self.inter)
        
        self.query = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
            nn.Dropout(0.3)
        )
        
        self.key = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
            nn.Dropout(0.3)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
            nn.Dropout(0.3)
        )
        
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        h = rearrange(x, 'b d c t -> b d t c')
        h_query = rearrange(self.query(h), 'b d t c -> b d c t')
        h_key = rearrange(self.key(h), 'b d t c -> b d c t')
        
        channel_query = self.pooling(h_query)
        channel_key = self.pooling(h_key)
        
        scaling = self.extract_sequence ** (1 / 2)
        
        channel_att = torch.einsum('b d c t, b d m t -> b d c m', channel_query, channel_key) / scaling
        
        channel_att_score = self.softmax(channel_att)
        
        output = torch.einsum('b d c t, b d c m -> b d c t', x, channel_att_score)
        output = rearrange(output, 'b d c t -> b d t c')
        output = self.projection(output)
        output = rearrange(output, 'b d t c -> b d c t')
        
        return output
        
        
# class PatchEmbedding(nn.Module):
#     def __init__(self, emb_size, num_channels):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Conv2d(1, 2, kernel_size=(1, 51)),
#             nn.BatchNorm2d(2),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(2, emb_size, kernel_size=(num_channels, 5), stride=(1, 5)),
#             Rearrange('b e (h) (w) -> b (h w) e')
#         )
        
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        
        
#     def forward(self, x):
#         b = x.size(0)
#         x= self.projection(x)
#         #cls_tokens = repeat(self.cls_tokenm '() n e -> b n e', b=b)
        
#         return x

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, sampling_rate, num_channels):
        super().__init__()
        kernel_size = int(sampling_rate * 0.16)
        pooling_kernel_size = int(sampling_rate * 0.3)
        pooling_stride_size = int(sampling_rate * 0.06)
        
        self.temporal_conv = Conv2dWithConstraint(1, emb_size, kernel_size=[1, kernel_size], padding='same', max_norm=2.)
        self.spatial_conv = Conv2dWithConstraint(emb_size, emb_size, kernel_size=[num_channels, 1], padding='valid', max_norm=2.)
        self.avg_pool = nn.AvgPool2d(kernel_size=[1, pooling_kernel_size], stride=[1, pooling_stride_size])
        self.bn = nn.BatchNorm2d(emb_size)
        self.rearrange = Rearrange('b e (h) (w) -> b (h w) e')
        
        
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-06))
        x = self.bn(x)
        x = self.rearrange(x)
        
        return x

    
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = x + res
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.5):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.quries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x, return_attention=False):
        queries = rearrange(self.quries(x), 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.num_heads)
        energy = torch.einsum('b h q d, b h k d -> b h q k', queries, keys)
        
        scaling = self.emb_size ** (1 / 2)
        att = self.softmax(energy / scaling)
        att = self.att_drop(att)
        output = torch.einsum('b h a l, b h l v -> b h a v', att, values)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.projection(output)
        if return_attention:
            return output, att
        else:
            return output
        
        
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, dropout=0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size)
        )
        
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=1, forward_expansion=1, dropout=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads),
                nn.Dropout(dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion),
                nn.Dropout(dropout)
            ))
        )
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__()
        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
        )
        
    
    def forward(self, x):
        output = self.cls_head(x)
        return output
    
    
class EEGTransformer(nn.Module):
    def __init__(self, temporal_size, num_channels, sampling_rate, emb_size=12, depth=1, is_gap: bool=False):
        super(EEGTransformer, self).__init__()
        
        self.ch_att = ResidualAdd(nn.Sequential(
                            nn.LayerNorm(temporal_size),
                            channel_attention(temporal_size=temporal_size, num_channels=num_channels),
                            nn.Dropout(0.5)
                        ))
        
        # self.patch_embedding = PatchEmbedding(emb_size, num_channels)
        self.patch_embedding = PatchEmbedding(emb_size, sampling_rate, num_channels)
        self.encoder = TransformerEncoder(depth, emb_size)
        self.clf_head = ClassificationHead(emb_size)
        
    
    def forward(self, x):
        x = self.ch_att(x)
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.clf_head(x)
        return x
    
    