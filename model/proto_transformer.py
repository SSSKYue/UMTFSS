import numbers
from telnetlib import SGA
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
import random
from model.ops.modules import MSDeformAttn
from model.positional_encoding import SinePositionalEncoding
from util.cluster import Cluster

class FFN(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

class ProtoCrossAtt(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.ass_drop   = nn.Dropout(0.1)

        self.drop_prob = 0.1


    def forward(self, q, k, v, supp_valid_mask=None):
        B, N, C = q.shape
        N_s = k.size(1)

        q = self.q_fc(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        k = self.k_fc(k).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        v = self.v_fc(v).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if supp_valid_mask is not None:
            supp_valid_mask = supp_valid_mask.unsqueeze(1).repeat(1, self.num_heads, 1) 

        attn = (q @ k.transpose(-2, -1)) * self.scale 
            
        if supp_valid_mask is not None:
            supp_valid_mask = supp_valid_mask.unsqueeze(-2).float() 
            supp_valid_mask = supp_valid_mask * -10000.0
            attn = attn + supp_valid_mask       

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class SuppSelfAttention(nn.Module):
    def __init__(self,
                 embed_dims=384,
                 num_heads=8,
                 num_layers=2,
                 num_points=9,
                 use_ffn=True,
                 dropout=0.1,
                 shot=1,
                 ):
        super(SuppSelfAttention, self).__init__()
        self.embed_dims=embed_dims
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.num_points=num_points
        self.use_ffn                = use_ffn
        self.feedforward_channels   = embed_dims*3
        self.dropout                = dropout
        self.shot                   = shot
        self.use_self               = True
        self.use_cluster            = True

        if self.use_cluster:
            ### clustering related args
            self.clusters_fg = 20
            self.clusters_bg = 50
            self.n_iter = 10
            self.ClusterTool = Cluster(self.n_iter)
        else:
            self.rand_fg_num = 300 * shot
            self.rand_bg_num = 300 * shot

        self.supp_self_layers = []
        self.layer_norms = []
        self.ffns = []
        for l_id in range(self.num_layers):
            if self.use_self:
                self.supp_self_layers.append(
                    MSDeformAttn(embed_dims, 1, 12 if embed_dims%12==0 else self.num_heads, num_points)
                )
            self.layer_norms.append(nn.LayerNorm(embed_dims))

            if self.use_ffn:
                self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                self.layer_norms.append(nn.LayerNorm(embed_dims))
        if self.use_self:
            self.supp_self_layers = nn.ModuleList(self.supp_self_layers)
        if self.use_ffn:
            self.ffns         = nn.ModuleList(self.ffns)
        self.layer_norms  = nn.ModuleList(self.layer_norms)
        self.positional_encoding = SinePositionalEncoding(embed_dims//2, normalize=True) 
        self.proj_drop  = nn.Dropout(dropout)
        # self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_supp_flatten_input(self, s_x, s_padding_mask, supp_mask):
        s_x_flatten_list = []
        supp_valid_mask = []
        supp_obj_mask = []

        supp_mask = F.interpolate(supp_mask, size=s_x.shape[-2:], mode='nearest').squeeze(1) 
        supp_mask = supp_mask.view(-1, self.shot, s_x.size(2), s_x.size(3))

        s_padding_mask = F.interpolate(s_padding_mask, size=s_x.shape[-2:], mode='nearest').squeeze(1)
        s_padding_mask = s_padding_mask.view(-1, self.shot, s_x.size(2), s_x.size(3))
        s_x = s_x.view(-1, self.shot, s_x.size(1), s_x.size(2), s_x.size(3))

        spatial_shape = [s_x.shape[-2:]]
        spatial_shape = torch.as_tensor(spatial_shape, dtype=torch.long, device=s_x.device) 

        for st_id in range(s_x.size(1)):
            supp_valid_mask_s = []
            supp_obj_mask_s = []
            for img_id in range(s_x.size(0)):
                supp_valid_mask_s.append(s_padding_mask[img_id, st_id, ...]==255)
                obj_mask = supp_mask[img_id, st_id, ...]==1
                if obj_mask.sum() == 0: # To avoid NaN
                    obj_mask[obj_mask.size(0)//2-1:obj_mask.size(0)//2+1, obj_mask.size(1)//2-1:obj_mask.size(1)//2+1] = True
                if (obj_mask==False).sum() == 0: # To avoid NaN
                    obj_mask[0, 0]   = False
                    obj_mask[-1, -1] = False 
                    obj_mask[0, -1]  = False
                    obj_mask[-1, 0]  = False
                supp_obj_mask_s.append(obj_mask)
            supp_valid_mask_s = torch.stack(supp_valid_mask_s, dim=0) 
            supp_valid_mask_s = supp_valid_mask_s.flatten(1)
            supp_valid_mask.append(supp_valid_mask_s)

            supp_obj_mask_s = torch.stack(supp_obj_mask_s, dim=0)
            supp_obj_mask_s = (supp_obj_mask_s==1).flatten(1) 
            supp_obj_mask.append(supp_obj_mask_s)
            
            s_x_s = s_x[:, st_id, ...]
            if len(s_x_s.shape) == 4:
                s_x_s = s_x_s.flatten(2).permute(0, 2, 1) 
            else:
                s_x_s = s_x_s.permute(0, 2, 1) 
            s_x_flatten_list.append(s_x_s)

        return s_x_flatten_list, supp_valid_mask, supp_obj_mask, spatial_shape

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
        return reference_points

    def pixel_sampling(self, s_x, supp_valid_mask, supp_mask):
        if self.training:
            scale_min = 0.6
            scale_max = 4.0 if self.shot==1 else 1.4
            sampling_scale = random.uniform(scale_min, scale_max)
            rand_fg_num = int(self.rand_fg_num*sampling_scale)
            rand_bg_num = int(self.rand_bg_num*sampling_scale)
        else:
            rand_fg_num = self.rand_fg_num
            rand_bg_num = self.rand_bg_num
        re_arrange_k = []
        re_arrange_valid_mask = []
        for b_id in range(s_x.size(0)):
            k_b = s_x[b_id]
            supp_mask_b = supp_mask[b_id]

            fg_k = k_b[supp_mask_b]
            bg_k = k_b[supp_mask_b==False]
            fg_valid = supp_valid_mask[b_id][supp_mask_b]
            bg_valid = supp_valid_mask[b_id][supp_mask_b==False]
            
            num_fg = fg_k.shape[0]
            num_bg = bg_k.shape[0]

            if (num_fg + num_bg) < (rand_fg_num + rand_bg_num):
                rest_num = rand_fg_num+rand_bg_num-num_fg-num_bg
                dim = fg_k.shape[1]
                rest_k = torch.zeros((rest_num, dim), dtype=fg_k.dtype, device=fg_k.device)
                rest_valid = torch.ones((rest_num), dtype=fg_valid.dtype, device=fg_valid.device)
                re_k = torch.cat([fg_k, bg_k, rest_k], dim=0)
                re_valid_mask = torch.cat([fg_valid, bg_valid, rest_valid], dim=0)

            elif num_fg<rand_fg_num:
                rest_num = rand_fg_num+rand_bg_num-num_fg
                bg_select_idx = torch.randperm(num_bg)[:rest_num]                
                re_k = torch.cat([fg_k, bg_k[bg_select_idx]], dim=0)
                re_valid_mask = torch.cat([fg_valid, bg_valid[bg_select_idx]], dim=0)

            elif num_bg<rand_bg_num:
                rest_num = rand_fg_num+rand_bg_num-num_bg
                fg_select_idx = torch.randperm(num_fg)[:rest_num]
                re_k = torch.cat([fg_k[fg_select_idx], bg_k], dim=0)
                re_valid_mask = torch.cat([fg_valid[fg_select_idx], bg_valid], dim=0)
                
            else:
                fg_select_idx = torch.randperm(num_fg)[:rand_fg_num]
                bg_select_idx = torch.randperm(num_bg)[:rand_bg_num]
                re_k = torch.cat([fg_k[fg_select_idx], bg_k[bg_select_idx]], dim=0)
                re_valid_mask = torch.cat([fg_valid[fg_select_idx], bg_valid[bg_select_idx]], dim=0)

            re_arrange_k.append(re_k)
            re_arrange_valid_mask.append(re_valid_mask)

        k = torch.stack(re_arrange_k, dim=0)
        re_arrange_valid_mask = torch.stack(re_arrange_valid_mask, dim=0)

        return k, re_arrange_valid_mask


    def cluster_group_lsh_kmeans(self, s_x, supp_valid_mask, supp_mask, use_valid=False):
        re_arrange_k = []
        re_arrange_valid_mask = []
        for b_id in range(s_x.size(0)):
            k_b = s_x[b_id] # [num_elem, c]
            supp_mask_b = supp_mask[b_id] # [num_elem]

            fg_k = k_b[supp_mask_b] # [num_fg, c]
            bg_k = k_b[supp_mask_b==False] # [num_bg, c]
            fg_valid = supp_valid_mask[b_id][supp_mask_b]
            bg_valid = supp_valid_mask[b_id][supp_mask_b==False]
            if not use_valid:
                fg_k = fg_k[fg_valid==0]
                bg_k = bg_k[bg_valid==0]
            
            num_fg = fg_k.shape[0]
            num_bg = bg_k.shape[0]

            s_grouped = []
            cluster_L = 0
            for keys, cluster_num, pixel_num in zip([fg_k, bg_k], [self.clusters_fg, self.clusters_bg], [num_fg, num_bg]):
                ### groups: L to clusters
                L, E = keys.shape
                # print('keys', keys.shape)
                if cluster_num == 1:
                    s_grouped.append(keys.mean(dim=0, keepdim=True))
                    cluster_L += 1
                elif pixel_num <= cluster_num:
                    cluster_L += pixel_num
                    s_grouped.append(keys)
                else:
                    cluster_L += cluster_num
                    with torch.no_grad():
                        proj, counts = self.ClusterTool(keys, cluster_num)
                        proj, counts = proj.to(keys.device), counts.unsqueeze(-1).to(keys.device)
                        proj = F.one_hot(proj, num_classes=cluster_num).float()
                    grouped = torch.mm(proj.transpose(0,1), keys) / (counts + 0.001)
                    s_grouped.append(grouped)
            
            padding_k = torch.zeros((self.clusters_fg + self.clusters_bg - cluster_L, E), dtype=s_grouped[0].dtype, device=s_grouped[0].device)
            s_grouped.append(padding_k)
            re_k = torch.cat(s_grouped, dim=0)
            valid_mask = torch.ones((self.clusters_fg + self.clusters_bg), dtype=s_grouped[0].dtype, device=s_grouped[0].device)
            valid_mask[:cluster_L] = 0
            re_arrange_k.append(re_k)
            re_arrange_valid_mask.append(valid_mask)

        k = torch.stack(re_arrange_k, dim=0)
        re_arrange_valid_mask = torch.stack(re_arrange_valid_mask, dim=0)

        return k, re_arrange_valid_mask

    def forward(self, s_x, s_padding_mask, supp_mask):
        s_x_list, s_valid_list, s_mask_list, s_spatial_shape= self.get_supp_flatten_input(s_x, s_padding_mask.clone(), supp_mask.clone())
        level_start_index_supp = torch.cat((s_spatial_shape.new_zeros((1, )), s_spatial_shape.prod(1).cumsum(0)[:-1])) 
        supp_ref_points = self.get_reference_points(s_spatial_shape, device=s_x_list[0].device)
        
        if self.use_self:
            for s, s_valid in zip(s_x_list, s_valid_list):
                s_valid_mask = s_valid.view(-1, s_spatial_shape[0][0], s_spatial_shape[0][1])
                supp_pos = self.positional_encoding(s_valid_mask)
                supp_pos = supp_pos.flatten(2).transpose(1, 2)
                ln_id = 0
                ffn_id = 0
                for l_id in range(self.num_layers):      
                    s =  s + self.proj_drop(self.supp_self_layers[l_id](s + supp_pos, supp_ref_points, s, s_spatial_shape, level_start_index_supp, s_valid))
                    s = self.layer_norms[ln_id](s)
                    ln_id += 1

                    if self.use_ffn:
                        s = self.ffns[ffn_id](s)
                        ffn_id += 1
                        s = self.layer_norms[ln_id](s)
                        ln_id += 1
        
        s_bs = torch.cat(s_x_list, 1)
        valid_mask_bs = torch.cat(s_valid_list, 1)
        mask_bs = torch.cat(s_mask_list, 1)
        if self.use_cluster:
            clusters, cluster_valid_mask = self.cluster_group_lsh_kmeans(s_bs, valid_mask_bs, mask_bs)
            clusters = F.normalize(clusters, dim=-1, p=2)
        else:
            clusters, cluster_valid_mask = self.pixel_sampling(s_bs, valid_mask_bs, mask_bs)
        return clusters, cluster_valid_mask


class ProtoTransformer(nn.Module):
    def __init__(self,
                 embed_dims=384, 
                 num_heads=8, 
                 num_layers=2,
                 num_levels=1,
                 num_points=9,
                 use_ffn=True,
                 dropout=0.1,
                 shot=1,
                 ):
        super(ProtoTransformer, self).__init__()
        self.embed_dims             = embed_dims
        self.num_heads              = num_heads
        self.num_layers             = num_layers
        self.num_levels             = num_levels
        self.num_points             = num_points
        self.use_ffn                = use_ffn
        self.feedforward_channels   = embed_dims*3
        self.dropout                = dropout
        self.shot                   = shot
        self.use_cross              = True
        self.use_self               = True
        self.use_self_supp          = True

        if self.use_cross:
            self.cross_layers = []
        self.qry_self_layers  = []
        self.layer_norms = []
        self.ffns = []
        for l_id in range(self.num_layers):
            if self.use_cross:
                self.cross_layers.append(
                    ProtoCrossAtt(embed_dims, num_heads=12 if embed_dims%12==0 else self.num_heads, attn_drop=self.dropout, proj_drop=self.dropout),
                )
                self.layer_norms.append(nn.LayerNorm(embed_dims))
                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))
            
            if self.use_self:
                self.qry_self_layers.append(
                    MSDeformAttn(embed_dims, num_levels, 12 if embed_dims%12==0 else self.num_heads, num_points)
                )
                self.layer_norms.append(nn.LayerNorm(embed_dims))

                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))
        
        self.self_supp_cluster = SuppSelfAttention(embed_dims=embed_dims, shot=self.shot, num_points=9)

        if self.use_cross: 
            self.cross_layers = nn.ModuleList(self.cross_layers)
        if self.use_self:
            self.qry_self_layers  = nn.ModuleList(self.qry_self_layers)
        if self.use_ffn:
            self.ffns         = nn.ModuleList(self.ffns)
        self.layer_norms  = nn.ModuleList(self.layer_norms)

        self.positional_encoding = SinePositionalEncoding(embed_dims//2, normalize=True)

        self.proj_drop  = nn.Dropout(dropout)

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
        return reference_points

    def get_qry_flatten_input(self, x, qry_masks):
        src_flatten = [] 
        qry_valid_masks_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []        
        for lvl in range(self.num_levels):   
            src = x[lvl]
            bs, c, h, w = src.shape
            
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).permute(0, 2, 1)
            src_flatten.append(src)

            if qry_masks is not None:
                qry_mask = qry_masks[lvl]
                qry_valid_mask = []
                qry_mask = F.interpolate(
                    qry_mask.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                for img_id in range(bs):
                    qry_valid_mask.append(qry_mask[img_id]==255)
                qry_valid_mask = torch.stack(qry_valid_mask, dim=0)
            else:
                qry_valid_mask = torch.zeros((bs, h, w))

            pos_embed = self.positional_encoding(qry_valid_mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_embed_flatten.append(pos_embed)

            qry_valid_masks_flatten.append(qry_valid_mask.flatten(1))

        src_flatten = torch.cat(src_flatten, 1)
        qry_valid_masks_flatten = torch.cat(qry_valid_masks_flatten, dim=1)
        pos_embed_flatten = torch.cat(pos_embed_flatten, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        return src_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index

    def transformer(self, x, qry_masks, k, v, cluster_valid_mask):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(qry_masks, list):
            qry_masks = [qry_masks.clone() for _ in range(self.num_levels)]
        assert len(x) == len(qry_masks) == self.num_levels
        bs, c = x[0].size()[:2]

        x_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index = self.get_qry_flatten_input(x, qry_masks)
        
        reference_points = self.get_reference_points(spatial_shapes, device=x_flatten.device)

        q = x_flatten
        pos = pos_embed_flatten

        ### qry self attention & cross attention
        ln_id = 0
        ffn_id = 0
        for l_id in range(self.num_layers):
            if self.use_self:
                q =  q + self.proj_drop(self.qry_self_layers[l_id](q + pos, reference_points, q, spatial_shapes, level_start_index, qry_valid_masks_flatten))
                q = self.layer_norms[ln_id](q)
                ln_id += 1
       
                if self.use_ffn:
                    q = self.ffns[ffn_id](q)
                    ffn_id += 1
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1

            if self.use_cross:
                cross_out = self.cross_layers[l_id](q, k, v, cluster_valid_mask)

                q = cross_out + q
                q = self.layer_norms[ln_id](q)
                ln_id += 1

                if self.use_ffn:
                    q = self.ffns[ffn_id](q)
                    ffn_id += 1
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1

        qry_feat = q.permute(0, 2, 1).contiguous() 
        h, w = spatial_shapes[0]
        qry_feat = qry_feat.view(bs, c, h, w)
        return qry_feat
            
    def forward(self, x, qry_masks, s_x, supp_mask, s_padding_mask):
        k, cluster_valid_mask = self.self_supp_cluster(s_x, s_padding_mask, supp_mask)

        v = k.clone()
        final_q_feat = self.transformer(x, qry_masks, k, v, cluster_valid_mask)
        return final_q_feat
