import torch
from torch import nn
import torch.nn.functional as F

from model.resnet import *
from model.loss import WeightedDiceLoss
from model.proto_transformer import ProtoTransformer
from model.ops.modules import MSDeformAttn
from model.backbone_utils import Backbone

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

class FSSModel(nn.Module):
    def __init__(self, layers=50, classes=2, shot=1, reduce_dim=384, criterion=WeightedDiceLoss(), with_transformer=True):
        super(FSSModel, self).__init__()
        assert layers in [50, 101]
        assert classes > 1 
        self.layers = layers
        self.criterion = criterion
        self.shot = shot
        self.with_transformer = with_transformer
        self.reduce_dim = reduce_dim

        self.print_params()

        in_fea_dim = 1024 + 512      
        drop_out = 0.5
        self.adjust_feature_supp = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        ) 
        self.adjust_feature_qry = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        )

        self.high_avg_pool = nn.Identity()

        prior_channel = 1
        self.qry_merge_feat = nn.Sequential(
                    nn.Conv2d(reduce_dim*2+prior_channel, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                )

        if self.with_transformer:
            self.supp_merge_feat = nn.Sequential(
                nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
            self.addtional_proj = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, bias=False)
            )
            self.transformer = ProtoTransformer(embed_dims=reduce_dim, shot=self.shot, num_points=9)
        else:
            self.merge_res = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )

        self.classifier = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),
                    nn.Conv2d(reduce_dim, classes, kernel_size=1)
                )

        qry_dim_scalar = 1
        self.pred_supp_qry_proj = nn.Sequential(
                nn.Conv2d(reduce_dim*qry_dim_scalar, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )

        self.init_weights()
        self.backbone = Backbone('resnet{}'.format(layers), train_backbone=False, return_interm_layers=True, dilation=[False, True, True])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()


    def print_params(self):
        repr_str = self.__class__.__name__
        repr_str += f'(backbone layers={self.layers}, '
        repr_str += f'reduce_dim={self.reduce_dim}, '
        repr_str += f'shot={self.shot}, '
        repr_str += f'with_transformer={self.with_transformer})'
        print(repr_str)
        return repr_str


    def forward(self, x, s_x, s_y, x_aug=None, y=None, padding_mask=None, s_padding_mask=None):
        '''
            x: [bs,3,h,w]
            y: [bs,h,w]
            padding_mask: [bs,h,w]
            s_x: [bs,shots,3,h,w]
            s_y: [bs,shots,h,w]
            s_padding_mask: [bs,shots,h,w] 
            x_aug: [bs,k,3,h,w], k: committee_size 
        '''
        batch_size, _, h, w = x.size()
        assert (h-1) % 8 == 0 and (w-1) % 8 == 0
        img_size = x.size()[-2:]
        if x_aug is not None:
            k = x_aug.shape[1]
            qs_x = torch.cat([x.unsqueeze(1), x_aug], dim=1)
            x = qs_x.view(-1, 3, *img_size)
        
        # backbone feature extraction
        qry_bcb_fts = self.backbone(x)
        supp_bcb_fts = self.backbone(s_x.view(-1, 3, *img_size)) 
        query_feat = torch.cat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
        supp_feat = torch.cat([supp_bcb_fts['1'], supp_bcb_fts['2']], dim=1)
        query_feat = self.adjust_feature_qry(query_feat)
        supp_feat = self.adjust_feature_supp(supp_feat) 
        fts_size = query_feat.shape[-2:]
        supp_mask = F.interpolate((s_y==1).view(-1, *img_size).float().unsqueeze(1), size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)

        # format support feature list for k shots
        supp_feat_list = []
        supp_mask_list = []
        r_supp_feat = supp_feat.view(batch_size, self.shot, -1, fts_size[0], fts_size[1])
        for st in range(self.shot):
            mask_s = (s_y[:,st,:,:]==1).float().unsqueeze(1)
            mask_s = F.interpolate(mask_s, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            tmp_supp_feat = r_supp_feat[:,st,...]
            supp_mask_list.append(mask_s)
            supp_feat_list.append(tmp_supp_feat)

        # support global feature extraction
        supp_global_feats = []
        for st in range(self.shot):
            supp_global_feats.append(Weighted_GAP(supp_feat_list[st], supp_mask_list[st]))
        global_supp_pp = supp_global_feats[0]
        if self.shot > 1:
            for i in range(1, len(supp_global_feats)):
                global_supp_pp += supp_global_feats[i]
            global_supp_pp /= len(supp_global_feats)
            multi_supp_pp = Weighted_GAP(supp_feat, supp_mask)
        else:
            multi_supp_pp = global_supp_pp

        supp_feat_high = supp_bcb_fts['3'].view(batch_size, -1, 2048, fts_size[0], fts_size[1])

        # prior generation
        query_feat_high = qry_bcb_fts['3']
        corr_query_mask = self.generate_prior(query_feat_high, supp_feat_high, s_y, fts_size)
            
        # feature mixing
        query_cat_feat = [query_feat, global_supp_pp.expand(-1, -1, fts_size[0], fts_size[1]), corr_query_mask]
        query_feat = self.qry_merge_feat(torch.cat(query_cat_feat, dim=1))


        if self.with_transformer:
            to_merge_fts = [supp_feat, multi_supp_pp.expand(-1, -1, fts_size[0], fts_size[1])]
            aug_supp_feat = torch.cat(to_merge_fts, dim=1)
            aug_supp_feat = self.supp_merge_feat(aug_supp_feat)

            fused_query_feat = self.transformer(query_feat, padding_mask.float(), aug_supp_feat, s_y.clone().float(), s_padding_mask.float())
        
        else:
            query_feat = self.merge_res(query_feat) + query_feat
            query_feat_list = [query_feat]
            fused_query_feat = query_feat.clone()


        # Output Part
        out = self.classifier(fused_query_feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
    
        if self.training:
            # calculate loss
            main_loss = self.criterion(out, y.long()) 
            pred = out.max(1)[1].detach()            
            return pred, main_loss
        else:
            return out


    def generate_prior(self, query_feat_high, supp_feat_high, s_y, fts_size):
        bsize, _, sp_sz, _ = query_feat_high.size()[:]
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for st in range(self.shot):
            tmp_mask = (s_y[:,st,:,:] == 1).float().unsqueeze(1)
            tmp_mask = F.interpolate(tmp_mask, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)

            tmp_supp_feat = supp_feat_high[:,st,...] * tmp_mask         
            q = self.high_avg_pool(query_feat_high.flatten(2).transpose(-2, -1)) 
            s = self.high_avg_pool(tmp_supp_feat.flatten(2).transpose(-2, -1)) 

            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1) 
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        return corr_query_mask