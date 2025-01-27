# -*- coding： utf-8 -*-
'''
@Time: 2023/12/4 15:54
@Author:YilanZhang
@Filename:model_acl.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F
import math
from models.sparsemax import Sparsemax
from models.mca import CrossTransformer


# ---- attention models for FDT
class Query_model(nn.Module):
    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='softmax', pool_type='sum'):
        '''
        ft_dim: feature dim of image patch or text token
        sd_dim: dim of FDT
        temperature: temperature for softmax or sparsemax
        att_func_type: attention normlization function type
        pool_type: pooling type for attention weights
        '''

        super().__init__()

        # activation
        assert att_func_type in ['softmax', 'sigmoid', 'sparsemax']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        elif self.att_func_type == 'sparsemax':
            self.att_activation = Sparsemax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature

        # map patch/text tokens to codebook (query) spaces
        # ---note that we donot use mapping for FDT

        self.q_map = nn.Sequential(
            nn.LayerNorm(ft_dim),
            nn.Linear(ft_dim, sd_dim),
            nn.GELU(),
            nn.LayerNorm(sd_dim),
            nn.Linear(sd_dim, sd_dim)
        )

    def forward(self, ft, sd, mask=None, return_token_att=False):

        '''
        Args:
            ft: [batch, token_num, ft_dim]
            sd: [FDT_num, sd_dim]
            mask: [batch, token_num]: mask for padded tokens.
            return_token_att: flag for returning attention weights before nomalization.
            used for visualizing FDT.
        Returns:

        '''

        # map image/text token to query space
        # ft = ft.transpose(1,3)  # [batch, token_num, ft_dim] token_num = 1
        B, H, W, C = ft.shape
        ft = ft.reshape(B, H * W, C)
        q = self.q_map(ft)  # bacth, token_num, dim

        k = sd  # code_num, sd_dim
        k = k.unsqueeze(0)  # [1, code_num, sd_dim]
        k = k.transpose(2, 1)  # [1,sd_dim, sd_num]

        # -----calculate inner dot
        inner_dot = torch.matmul(q, k)  # [bacth, token_num, code_num]

        if return_token_att:  # cosine sim
            token_att = inner_dot

        inner_dot = inner_dot / math.sqrt(self.att_dim)  # scale dot norm

        if mask is not None:  # mask paded tokens

            assert mask.shape == q.shape[:2]
            mask = (mask == 0) * 1  # 0 --> 1, inf --> 0

            inner_dot = inner_dot * mask.unsqueeze(-1)  # sigmod(-inf) = 0, softmax(-inf) = 0

            if return_token_att:  # if has pad, return maksed
                token_att = inner_dot

        # temptural norm
        inner_dot = inner_dot / self.temperature  # [bacth, token_num, code_num]

        # pooling
        if self.pool_type == 'sum':
            inner_dot = inner_dot.sum(1)  # mean poolings
        elif self.pool_type == 'mean':
            inner_dot = inner_dot.mean(1)
        else:
            inner_dot = inner_dot.max(1)[0]

        # ----get attention weights
        att_weight = self.att_activation(inner_dot)  # normaliztion


        att_ft = att_weight @ sd  # [bacth, dictory_size] * [dictory_size, dim]  ---> [bacth, sd_num, dim]

        if self.att_func_type == 'sigmoid':
            att_ft = att_ft / att_weight.sum(dim=-1, keepdim=True)

        if return_token_att:
            return token_att, att_ft, sd
        return att_weight, att_ft, sd




class model_multiskin(nn.Module):
    def __init__(self,encoder,dim=256,num_classes=8,shared=False,args=None):
        super(model_multiskin, self).__init__()
        self.args = args
        self.shared = shared
        encoder_ = nn.Sequential(*list(encoder.children())[:-2])
        if self.shared:
            self.encoder = encoder_
            # 去掉最后两层
        else:
            self.encoder_cli = encoder_
            self.encoder_derm = encoder_

        dim_latent = 768

        if self.args.fdt:
            # learnable FDT
            self.space_dict = nn.Parameter(torch.randn(256, dim_latent))
            self.cli_query_model = Query_model(ft_dim=dim_latent, sd_dim=dim_latent, pool_type='mean', att_func_type='softmax')
            self.derm_query_model = Query_model(ft_dim=dim_latent, sd_dim=dim_latent, pool_type='mean', att_func_type='softmax')

        else:

            self.pooling = encoder.head.global_pool

        self.fusion = CrossTransformer(x_dim=dim_latent,c_dim=dim_latent,depth=2,num_heads=4)

        self.fc_head = nn.Sequential(
            nn.Linear(dim_latent * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(dim, num_classes)
        )
        self.fc_head_2 = nn.Sequential(
            nn.Linear(dim_latent * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(dim, 2)
        )


    def forward(self, x,eval=False):
        if not isinstance(x, list):
            raise ValueError('x must be a list of two tensors')
        if self.shared:
            feat1 = self.encoder(x[0])
            feat2 = self.encoder(x[1])
        else:
            feat1 = self.encoder_cli(x[0])
            feat2 = self.encoder_derm(x[1])

        if self.args.fdt:
            _,feat1,_ = self.cli_query_model(feat1, self.space_dict)
            _,feat2,_ = self.derm_query_model(feat2, self.space_dict)
        else:
            feat1 = self.pooling(feat1.transpose(1, 2))
            feat2 = self.pooling(feat2.transpose(1, 2))
            feat1 = feat1.view(feat1.shape[0], -1)
            feat2 = feat2.view(feat2.shape[0], -1)

        cli_cross_attention_feat, der_cross_attention_feat = self.fusion(feat1, feat2)
        feat = torch.cat((cli_cross_attention_feat,der_cross_attention_feat), dim=1)
        logits = self.fc_head(feat)
        logits_2 = self.fc_head_2(feat)

        if not eval:
            feat1_head = F.normalize(feat1)
            feat2_head = F.normalize(feat2)
            return [logits,logits_2], [feat1_head, feat2_head]
        return [logits,logits_2]



def build_model_multiskin(args):
    encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    model = model_multiskin(encoder,num_classes=args.class_num,shared=args.shared,args=args)
    return model