# -*- coding： utf-8 -*-
'''
@Time: 2022/5/17 19:35
@Author:YilanZhang
@Filename:loss.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''


import torch
import torch.nn as nn

class Bal_MultiModal_CL(nn.Module):
    def __init__(self,num_classes=8,temperature=0.01):
        super(Bal_MultiModal_CL, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, features1,features2,targets):
        device = (torch.device('cuda') if features1.is_cuda else torch.device('cpu'))
        batch_size = features1.shape[0]
        targets = targets.contiguous().view(-1, 1)

        targets = torch.cat([targets.repeat(2,1)],dim=0)
        batch_cls_count = torch.eye(self.num_classes).to(device)
        batch_cls_count = batch_cls_count[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets,targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )  # 用0的元素替换1，index为torch.arange(batch_size * 2).view(-1, 1).to(device),这一步的用意在于去掉元素自身
        mask = mask * logits_mask

        #features
        features1 = torch.cat(torch.unbind(features1, dim=1), dim=0)
        features2 = torch.cat(torch.unbind(features2, dim=1), dim=0)
        logits = torch.mm(features1, features2.T) / self.temperature

        #for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        #class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(2 * batch_size, 2 * batch_size ) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits_sum)  # [batch_size*2,batch_size*2+class_num]
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # (batch_size*2)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss