import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import os
from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, knn_gather
from models.pointnet2_utils import PointNetFeaturePropagation
from .davit import DaViT

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel, group_size):
        super().__init__()
        self.factor = 1
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, int(128*self.factor), 1),
            nn.BatchNorm1d(int(128*self.factor)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(128*self.factor), int(256*self.factor), 1)
        )
        #self.color_conv = nn.Sequential(
        #    nn.Conv1d(3, int(128*self.factor), 1),
        #    nn.BatchNorm1d(int(128*self.factor)),
        #    nn.ReLU(inplace=True),
        #    nn.Conv1d(int(128*self.factor), int(256*self.factor), 1)
        #)
        self.second_conv = nn.Sequential(
            nn.Conv1d(int(512*self.factor), int(512*self.factor), 1),
            nn.BatchNorm1d(int(512*self.factor)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(512*self.factor), self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , d = point_groups.shape
        if d == 6:
            points = point_groups[:,:,:,:3].reshape(bs * g, n, 3)
            color = point_groups[:,:,:,3:].reshape(bs * g, n, 3)

        else:
            points = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(points.transpose(2,1))  # BG 256 n
        #if d == 6:
        #    feature += self.color_conv(color.transpose(2,1))
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

    def forward_seg(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims, group_size=config.group_size)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, neighborhood, center, bool_masked_pos, noaug = False):

        group_input_tokens = self.encoder(neighborhood)  #  B G C
        batch_size, seq_len, C = group_input_tokens.size()
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        # prediction head
        self.increase_dim = nn.Sequential(
            #nn.Conv1d(self.trans_dim*2, 1024, 1),
            #nn.ReLU(),
            #nn.Conv1d(1024, 1024, 1),
            #nn.ReLU(),
            #nn.Conv1d(1024, 1024, 1),
            #nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type =="pytorchcdl2":
            self.loss_func = chamfer_distance_wrapper
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, neighborhood, center, mask, vis = False, **kwargs):
        x_vis = self.MAE_encoder(neighborhood, center, mask)
        B,_,C = x_vis.shape # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, self.group_size, 3)  #  B *Ntokens, npoints x3
        gt_points = neighborhood[mask].reshape(B*M,self.group_size, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis: # assume batch size 1
            assert neighborhood.shape[0] == 1
            ntokens = neighborhood[mask].shape[0]
            npoints = neighborhood[mask].shape[1]
            masked_points = neighborhood[mask] + center[mask].unsqueeze(1) # ntokens x npoints x 3 + ntokens x 1 x 3
            unmasked_points = neighborhood[~mask] + center[~mask].unsqueeze(1) # ntokens x npoints x 3 + ntokens x 1 x 3
            rebuilt_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2) # 1 x ntokens x npoints * 3
            rebuilt_points = rebuilt_points.reshape(ntokens, npoints, 3) # ntokens x npoints x 3 
            rebuilt_points = rebuilt_points + center[mask].unsqueeze(1)
            full_original = torch.cat([masked_points, unmasked_points], dim=0)
            full_rebuilt = torch.cat([unmasked_points, rebuilt_points], dim=0)
           
            return full_original, full_rebuilt, unmasked_points, rebuilt_points, masked_points
        else:
            return loss1

    def load_model_from_ckpt(self, ckpt_path, freeze_encoder=None, logger=None, remove_increase_dim=False):
        if not os.path.exists(ckpt_path):
            raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
        print_log(f'Loading weights from {ckpt_path}...', logger = logger )

        # load state dict
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # parameter resume of base model
        if state_dict.get('model') is not None:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
        elif state_dict.get('base_model') is not None:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        else:
            raise RuntimeError('mismatch of ckpt weight')
        #print(base_ckpt.keys())
        if remove_increase_dim:
            print_log("REMOVED INCREASE DIM", logger=logger)
            del base_ckpt["increase_dim.0.weight"]
            del base_ckpt["increase_dim.0.bias"]
        self.load_state_dict(base_ckpt, strict = False)

        epoch = -1
        if state_dict.get('epoch') is not None:
            epoch = state_dict['epoch']
        if state_dict.get('metrics') is not None:
            metrics = state_dict['metrics']
            if not isinstance(metrics, dict):
                metrics = metrics.state_dict()
        else:
            metrics = 'No Metrics'
        print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)

def chamfer_distance_wrapper(x,y):
    return chamfer_distance(x,y)[0]

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.task = args.task

        # shared encoder
        self.encoder = Encoder(encoder_channel=self.encoder_dims, group_size=config.group_size)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), # 3 always even with color
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # task specific layers
        if self.task == "regression" or self.task == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)
            if self.task == "cls":
                self.cls_head_finetune = nn.Sequential(
                    nn.Linear(self.trans_dim * 2, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, self.cls_dim)
                )
            if self.task == "regression":
              self.cls_head_finetune = nn.Sequential(nn.Linear(self.trans_dim * 2, self.cls_dim))
              self.apply(self._init_weights)

        elif self.task == "segmentation" or self.task == "offset":
            num_input_features = config.use_feature_prop * 1024 + config.use_token_features * 1152 + config.use_global_features * 1152 * 2
            self.lossweight = torch.tensor([1, 2]).float().cuda()

            self.use_feature_prop = config.use_feature_prop
            self.use_token_features = config.use_token_features 
            self.use_global_features = config.use_global_features
            if self.use_feature_prop:
                self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024])
            self.convs1 = nn.Conv1d(num_input_features, 512, 1) #4480 3328 2176 1152
            self.dp1 = nn.Dropout(0)
            self.convs2 = nn.Conv1d(512, 256, 1)
            self.convs3 = nn.Conv1d(256 + 3, self.cls_dim, 1)
            self.bns1 = nn.BatchNorm1d(512)
            self.bns2 = nn.BatchNorm1d(256)
            self.relu = nn.ReLU()
        self.build_loss_func(config.label_smoothing if self.task == "cls" else None)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        

    def build_loss_func(self, label_smoothing=None):
        if self.task == "cls":
            self.loss_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif self.task == "regression":
            self.loss_ce = torch.nn.MSELoss()
        elif self.task == "segmentation":
            self.loss_ce = self.seg_loss
        elif self.task == "offset":
            self.loss_ce = self.offset_loss
    
    def offset_loss(self, prediction, gt, get_l1=True):
        prediction = prediction.contiguous().view(-1, 3)
        gt = gt.contiguous().view(-1, 3)
        mask = gt[:, 0] != 14985
        prediction, gt = prediction[mask], gt[mask]
        if get_l1:
            return F.l1_loss(prediction, gt) * 3
        return F.pairwise_distance(prediction, gt).mean()


    def seg_loss(self, prediction, label):
        prediction = prediction.contiguous().view(-1, 2)
        label = label.view(-1, 1)[:, 0].long()
        return F.nll_loss(prediction, label, weight=self.lossweight)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path, freeze_encoder=False, logger=None):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if freeze_encoder:
                allpars = int(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000)
                for i, param in enumerate(self.blocks.parameters()):
                    param.requires_grad = False
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.pos_embed.parameters():
                    param.requires_grad = False
                npars = int(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000)
                print_log(f"Parmeters remaining {npars} / {allpars} * 1000", logger=logger)

            if incompatible.missing_keys:
                print_log('missing_keys', logger=logger)
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger=logger
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger=logger)
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger=logger
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger=logger)
        else:
            print_log('Training from scratch!!!', logger=logger)
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, *args):
        if self.task == "regression" or self.task == "cls":
            return self.cls_forward(*args)
        elif self.task == "segmentation" or self.task == "offset":
            return self.pointwise_forward(*args)

    def cls_forward(self, neighborhood, center):
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
    
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret

    def pointwise_forward(self, neighborhood, center, pts, idx):
        B, G, P, C = neighborhood.shape
        N = pts.shape[1]
        x = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)
        # transformer
        feature_list = self.blocks.forward_seg(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #B 3*384 G
        generated_features = []
        # these are global features and equal for every point
        if self.use_global_features:
            x_max_feature = torch.max(x,2)[0].view(B, -1).unsqueeze(-1).repeat(1, 1, N).permute(0,2,1)
            x_avg_feature = torch.mean(x,2).view(B, -1).unsqueeze(-1).repeat(1, 1, N).permute(0,2,1)
            generated_features.append(x_max_feature.permute(0,2,1))
            generated_features.append(x_avg_feature.permute(0,2,1))
        if self.use_token_features:
            _, idx, _ = knn_points(pts, center, K=1)
            feature = knn_gather(x.permute(0,2,1), idx)[:,:, 0,:]
            generated_features.append(feature.permute(0,2,1))
        if self.use_feature_prop:
            f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x) # B x 1024 x N
            generated_features.append(f_level_0)
            
        x = torch.cat((generated_features), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(torch.cat((x, pts.permute(0,2,1)), 1))
        if self.task == "segmentation":
            x = F.log_softmax(x, dim=2)
        #assert not torch.any(torch.isnan(x))
        x = x.permute(0, 2, 1)
        return x


