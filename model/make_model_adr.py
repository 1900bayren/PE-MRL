import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, \
    deit_small_patch16_224_TransReID, vit_large_patch16_224_TransReID, vit_large_patch32_224_TransReID, \
    deit_large_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from config import cfg
from .backbones.vit_pytorch import trunc_normal_

# === NEW: 物理先验监督头 ===
import torch.nn as nn
import torch.nn.functional as F

# 引入物理先验
class PhysicalPriorHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, p_drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=p_drop)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.out_features = out_dim
        # kaiming 初始化
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

# === 引入adr ===
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None
def grl(x, lam): return GradReverse.apply(x, lam)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=512, num_domains=2, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(True), nn.Dropout(p_drop),
            nn.Linear(hidden, num_domains))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)




'''
## 作用：特征通道/切片重排（用于JPM局部分支的分段与打散），增强局部特征多样性与鲁棒性。
## 参数：
##  - features: 输入特征张量 (B, N, D)
##  - shift: 重排的偏移量
##  - group: 分组数量
##  - begin: 开始索引位置
## 返回：重排后的特征张量
'''
def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


'''
## 作用：为网络层进行Kaiming初始化（又称He初始化），使得卷积层或全连接层在ReLU激活下保持输出方差稳定，缓解梯度消失或爆炸。
## 参数：
##  - m: 待初始化的层模块
'''


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


'''
## [函数] weights_init_classifier(m)
## 作用：分类层初始化（小方差），避免early over-confidence。
## 参数：
##  - m: 线性分类层
'''


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


'''
ResNet50骨干网络，含BNNeck与多种ID损失头
'''


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    '''
    训练/测试模式下提取全局特征并分类
    '''

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    """
    构建基于Transformer的深度学习模型，主要用于行人重识别(ReID)等任务
    集成了特征提取、瓶颈层和分类头，支持多种损失函数和预训练配置
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        """
        初始化模型结构

        参数:
            num_classes (int): 分类任务的类别数量
            camera_num (int): 相机数量，用于SIE(Spatial Information Embedding)
            view_num (int): 视角数量，用于SIE
            cfg: 配置对象，包含模型各种参数设置
            factory: Transformer模型工厂，用于创建不同类型的Transformer骨干网络
        """
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        # 根据配置决定是否使用相机和视角信息进行空间信息嵌入(SIE)
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # 创建Transformer Backbone
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,  # 输入图像尺寸
            sie_xishu=cfg.MODEL.SIE_COE,  # SIE系数
            camera=camera_num,  # 相机数量
            view=view_num,  # 视角数量
            stride_size=cfg.MODEL.STRIDE_SIZE,  # 步长大小
            drop_path_rate=cfg.MODEL.DROP_PATH,  # 路径丢弃率
            drop_rate=cfg.MODEL.DROP_OUT,  # 丢弃率
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE  # 注意力丢弃率
        )
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层，将特征图转为特征向量

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        """
        前向传播函数

        参数:
            x: 输入图像张量
            label: 标签，训练时使用
            cam_label: 相机标签，用于SIE
            view_label: 视角标签，用于SIE

        返回:
            训练时返回分类分数和全局特征；测试时返回特征向量
        """
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        """加载预训练模型参数，处理分布式训练的参数名前缀"""
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # 移除参数名中的'module.'前缀，适应非分布式加载
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print(f'Loading pretrained model from {trained_path}')

    def load_param_finetune(self, model_path):
        """加载模型参数用于微调，直接复制参数"""
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print(f'Loading pretrained model for finetuning from {model_path}')


class build_transformer_local(nn.Module):
    """
    构建基于Transformer的局部特征增强模型，适用于需要细粒度特征分析的任务（如行人重识别）
    相比基础Transformer模型，增加了局部特征分支、实例提示学习(IPIL)等模块，支持多尺度特征融合
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        """
        初始化模型结构，配置全局与局部特征分支、分类器及各类参数

        参数:
            num_classes (int): 分类任务的类别数量
            camera_num (int): 相机数量，用于空间信息嵌入(SIE)
            view_num (int): 视角数量，用于SIE
            cfg: 配置对象，包含模型各类参数设置
            factory: Transformer模型工厂，用于创建backbone
            rearrange (bool): 是否对特征进行重排列（用于数据增强）
        """
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768  # 默认特征维度
        self.specific_bn = cfg.MODEL.SPECIFIC_BN
        # 初始化零向量token（用于实例提示学习）
        self.zero_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        trunc_normal_(self.zero_token, std=.02)  # 使用截断正态分布初始化

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # 物理先验监督头
        self.use_phys = getattr(cfg.MODEL, "USE_PHYS_PRIOR", False)
        if self.use_phys:
            self.prior_dim_ir = getattr(cfg.MODEL, "PRIOR_DIM_IR", 36)
            self.prior_dim_vis = getattr(cfg.MODEL, "PRIOR_DIM_VIS", 35)
            self.ir_head = PhysicalPriorHead(self.in_planes, self.prior_dim_ir, hidden=1024)
            self.vis_head = PhysicalPriorHead(self.in_planes, self.prior_dim_vis, hidden=1024)
        # __init__ 里
        self.use_proxy_adr = getattr(cfg.MODEL, "USE_PROXY_ADR", False)
        if self.use_proxy_adr:
            self.proxy_num_domains = int(getattr(cfg.MODEL, "PROXY_NUM_DOMAINS", 4))  # 4 桶示例
            self.domain_disc_proxy = nn.Sequential(
                nn.Linear(self.in_planes, 512), nn.ReLU(True), nn.Dropout(0.1),
                nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.1),
                nn.Linear(512, self.proxy_num_domains)
            )
            for m in self.domain_disc_proxy.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        # ADR
        self.use_adr = getattr(cfg.MODEL, "USE_ADR", True)
        if self.use_adr:
            self.num_domains = getattr(cfg.MODEL, "NUM_DOMAINS", 2)
            # print("num_domains",self.num_domains)
            self.domain_disc = DomainDiscriminator(self.in_planes, hidden=512, num_domains=self.num_domains)

        # 创建Transformer backbone（支持局部特征提取、实例提示等功能）
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,  # SIE系数
            local_feature=cfg.MODEL.JPM,  # 是否启用局部特征
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            num_tokens=cfg.MODEL.NUM_TOKEN,  # token数量
            use_prompt=cfg.MODEL.USE_PROMPT,  # 是否使用提示学习
            num_instance_prompt_tokens=cfg.MODEL.NUM_INS_PMT_TOKEN,  # 实例提示token数量
            size_instance_prompt_bank=cfg.MODEL.SIZE_INS_PMT_BANK,  # 实例提示库大小
            use_instance_prompt=cfg.MODEL.USE_INS_PROMPT  # 是否使用实例提示
        )

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        # 复制骨干网络的最后一个块和归一化层，构建局部特征处理分支
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        # 根据配置添加第三个处理块（用于实例提示学习）在baseline基础上添加
        if cfg.MODEL.IPIL and not cfg.MODEL.IPIL_SIMPLE:
            self.b3 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE  # 身份损失函数类型
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            # 默认使用多个线性分类器（对应全局和局部特征分支）
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)
            self.classifiers_ins_prompts = nn.ModuleList()
            for i in range(10):
                classifier_ins_prompts = nn.Linear(self.in_planes, self.num_classes, bias=False)
                classifier_ins_prompts.apply(weights_init_classifier)
                self.classifiers_ins_prompts.append(classifier_ins_prompts)

        # 全局和局部特征的瓶颈层（BatchNorm1d）
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        # 为不同模态（如红外/可见光）配置专属BN层 在Baseline 基础上添加
        if self.specific_bn:
            self.bottleneck_sub = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub.bias.requires_grad_(False)
            self.bottleneck_sub.apply(weights_init_kaiming)
            self.bottleneck_sub_1 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub_1.bias.requires_grad_(False)
            self.bottleneck_sub_1.apply(weights_init_kaiming)
            self.bottleneck_sub_2 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub_2.bias.requires_grad_(False)
            self.bottleneck_sub_2.apply(weights_init_kaiming)
            self.bottleneck_sub_3 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub_3.bias.requires_grad_(False)
            self.bottleneck_sub_3.apply(weights_init_kaiming)
            self.bottleneck_sub_4 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub_4.bias.requires_grad_(False)
            self.bottleneck_sub_4.apply(weights_init_kaiming)

        self.bottlenecks_ins_prompts = nn.ModuleList()
        for i in range(10):
            bottleneck_ins_prompts = nn.BatchNorm1d(self.in_planes)
            bottleneck_ins_prompts.apply(weights_init_kaiming)
            self.bottlenecks_ins_prompts.append(bottleneck_ins_prompts)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def _apply_proxy_feat(self, feat: torch.Tensor, proxy_ids: torch.Tensor) -> torch.Tensor:
        """
        对 BNNeck 后的 feat 施加代理域扰动（分桶），保持可微。
        feat: [B, D], proxy_ids: [B] in {0..S-1}
        桶设计（示例可改）：
          0: identity（不变）
          1: scale-jitter（通道尺度抖动）
          2: gaussian-noise（高斯噪声）
          3: dropout-mask（特征维随机丢弃）
        """
        out = feat.clone()
        B, D = feat.shape
        # 逐桶处理，避免 for b in B
        for k in range(int(proxy_ids.max().item()) + 1):
            mask = (proxy_ids == k)
            if not mask.any():
                continue
            f = feat[mask]
            if k == 0:
                y = f
            elif k == 1:
                # 轻微尺度抖动，均值 1，方差 0.2
                scale = 1.0 + 0.2 * torch.randn_like(f)
                y = f * scale
            elif k == 2:
                # 加性高斯噪声
                y = f + 0.1 * torch.randn_like(f)
            elif k == 3:
                # 特征维 dropout（可微，反向正常传播）
                keep = (torch.rand_like(f) > 0.1).float()
                y = f * keep
            else:
                y = f
            out[mask] = y
        return out

    # def forward(self, x, label=None, cam_label= torch.tensor([0]).cuda(), view_label=None, modality_flag=torch.tensor([0]).cuda()):  # label is unused if self.cos_layer == 'no'
    def forward(self, x, label=None, cam_label=None, view_label=None,
                modality_flag=None, lam_adv: float=0.0):  # label is unused if self.cos_layer == 'no'
        """
        前向传播函数，同时处理全局特征和局部特征，支持训练和测试模式
        参数:
            x: 输入图像张量
            label: 标签（训练时使用）
            cam_label: 相机标签（用于SIE）
            view_label: 视角标签（用于SIE）
            modality_flag: 模态标签（区分不同数据类型，如红外/可见光）
        返回:
            训练时返回多分支分类分数和特征；测试时返回融合后的特征向量
        """
        # 根据配置提取骨干网络输出（支持TSNE可视化和实例提示学习）
        if cfg.TEST.TSNE:
            features, X_tsne_1, X_tsne_2 = self.base(x, cam_label=cam_label, view_label=view_label,
                                                     modality_flag=modality_flag)
        elif cfg.MODEL.USE_INS_PROMPT and cfg.MODEL.IPIL:
            features, ins_prompts = self.base(x, cam_label=cam_label, view_label=view_label,
                                              modality_flag=modality_flag)
        else:
            features = self.base(x, cam_label=cam_label, view_label=view_label, modality_flag=modality_flag)

        '''if cfg.MODEL.USE_INS_PROMPT and cfg.MODEL.IPIL:
            features, ins_prompts = self.base(x, cam_label=cam_label, view_label=view_label, modality_flag=modality_flag)
        else:
            features = self.base(x, cam_label=cam_label, view_label=view_label, modality_flag=modality_flag)
        '''
        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768] 经过最后一个Transformer块和归一化
        global_feat = b1_feat[:, 0]  # 取cls token作为全局特征

        # JPM branch Joint Part Modeling
        feature_length = features.size(1) - 1  # 排除cls token后的特征长度
        patch_length = feature_length // self.divide_length  # 每个局部特征的长度
        token = features[:, 0:1]  # 提取cls token

        # 对特征进行重排列（数据增强）
        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]  # 排除cls token，保留patch特征
        # lf_1 local feature
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        # 对不同模态特征使用专属BN层
        if self.specific_bn:
            # 按模态分割特征
            global_feat_ir = global_feat[modality_flag == 0]
            global_feat_rgb = global_feat[modality_flag == 1]
            local_feat_1_ir = local_feat_1[modality_flag == 0]
            local_feat_1_rgb = local_feat_1[modality_flag == 1]
            local_feat_2_ir = local_feat_2[modality_flag == 0]
            local_feat_2_rgb = local_feat_2[modality_flag == 1]
            local_feat_3_ir = local_feat_3[modality_flag == 0]
            local_feat_3_rgb = local_feat_3[modality_flag == 1]
            local_feat_4_ir = local_feat_4[modality_flag == 0]
            local_feat_4_rgb = local_feat_4[modality_flag == 1]
            # 分别通过专属BN层
            feat_ir = self.bottleneck_sub(global_feat_ir)
            feat_rgb = self.bottleneck(global_feat_rgb)
            local_feat_1_ir_bn = self.bottleneck_sub_1(local_feat_1_ir)
            local_feat_2_ir_bn = self.bottleneck_sub_2(local_feat_2_ir)
            local_feat_3_ir_bn = self.bottleneck_sub_3(local_feat_3_ir)
            local_feat_4_ir_bn = self.bottleneck_sub_4(local_feat_4_ir)
            local_feat_1_rgb_bn = self.bottleneck_1(local_feat_1_rgb)
            local_feat_2_rgb_bn = self.bottleneck_2(local_feat_2_rgb)
            local_feat_3_rgb_bn = self.bottleneck_3(local_feat_3_rgb)
            local_feat_4_rgb_bn = self.bottleneck_4(local_feat_4_rgb)
            # 拼接不同模态的特征
            feat = torch.cat((feat_ir, feat_rgb), dim=0)
            local_feat_1_bn = torch.cat((local_feat_1_ir_bn, local_feat_1_rgb_bn), dim=0)
            local_feat_2_bn = torch.cat((local_feat_2_ir_bn, local_feat_2_rgb_bn), dim=0)
            local_feat_3_bn = torch.cat((local_feat_3_ir_bn, local_feat_3_rgb_bn), dim=0)
            local_feat_4_bn = torch.cat((local_feat_4_ir_bn, local_feat_4_rgb_bn), dim=0)

        else:
            feat = self.bottleneck(global_feat)
            local_feat_1_bn = self.bottleneck_1(local_feat_1)
            local_feat_2_bn = self.bottleneck_2(local_feat_2)
            local_feat_3_bn = self.bottleneck_3(local_feat_3)
            local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:


            if self.specific_bn:
                # 拼接不同模态的原始特征（未经过BN层），用于后续损失计算
                global_feat = torch.cat((global_feat_ir, global_feat_rgb), dim=0)
                local_feat_1 = torch.cat((local_feat_1_ir, local_feat_1_rgb), dim=0)
                local_feat_2 = torch.cat((local_feat_2_ir, local_feat_2_rgb), dim=0)
                local_feat_3 = torch.cat((local_feat_3_ir, local_feat_3_rgb), dim=0)
                local_feat_4 = torch.cat((local_feat_4_ir, local_feat_4_rgb), dim=0)
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                # 对全局特征计算分类分数（这些损失函数需要标签参与计算）
                cls_score = self.classifier(feat, label)
            else:
                # 对全局和局部特征分别计算分类分数（普通线性分类器）
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)

            # (b) ★ DANN: 域对抗正则（camera_id 作为域标签）
            dom_logits = None
            # print("USE_ADR: ", getattr(cfg.MODEL, "USE_ADR", False))
            if getattr(cfg.MODEL, "USE_ADR", False) and lam_adv > 0.0:
                # 注意：这里不要用 modality_flag，当域标签用 IR/VIS 会抹掉模态判别性
                # 走 BN 后的 feat 更稳（也可在 global_feat 上并一条轻支路）
                rev_feat = GradReverse.apply(feat, lam_adv)
                dom_logits = self.domain_disc(rev_feat)  # 形状 [B, num_cameras]

            # (b2) ★ 代理域(Proxy) 对抗：完全在特征层做，不改输入
            dom_proxy_logits = None
            proxy_ids = None
            # print("USE_PROXY_ADR: ", getattr(cfg.MODEL, "USE_PROXY_ADR", False))
            if getattr(cfg.MODEL, "USE_PROXY_ADR", False) and lam_adv > 0.0 and self.training:
                B, D = feat.shape
                # 为当前 batch 随机采样代理域标签（S=PROXY_NUM_DOMAINS）
                S = int(getattr(cfg.MODEL, "PROXY_NUM_DOMAINS", 4))
                proxy_ids = torch.randint(low=0, high=S, size=(B,), device=feat.device, dtype=torch.long)
                # 对 feat 施加特征级扰动，得到代理域特征
                feat_proxy = self._apply_proxy_feat(feat, proxy_ids)
                # 经 GRL 到代理域判别器
                dom_proxy_logits = self.domain_disc_proxy(GradReverse.apply(feat_proxy, lam_adv))
            if cfg.MODEL.USE_INS_PROMPT and cfg.MODEL.IPIL:  # 如果启用实例提示学习(IPIL) 在Baseline TransReID基础上添加
                # ins_prompts_feats = [self.bottleneck_ins_prompts(ins_prompt) for ins_prompt in ins_prompts]
                # ins_prompts_cls_score = [self.classifier_ins_prompts(ins_prompts_feat) for ins_prompts_feat in ins_prompts_feats]
                # zero_token = self.zero_token.expand(features.size(0), -1, -1)
                # b_ins_prompts_feats = [self.b3(torch.cat((zero_token, ins_prompt), dim=1)) for ins_prompt in ins_prompts]
                # ins_prompts_feats = [b_ins_prompts_feat[:, 0, :] for b_ins_prompts_feat in b_ins_prompts_feats]
                # 处理实例提示特征
                if cfg.MODEL.IPIL_SIMPLE:
                    for i in range(10):  # 简单模式：直接对实例提示特征应用瓶颈层
                        ins_prompts[i] = self.bottlenecks_ins_prompts[i](ins_prompts[i])
                else:
                    zero_token = self.zero_token.expand(features.size(0), -1, -1)
                    b_ins_prompts_feats = [self.b3(torch.cat((zero_token, ins_prompt), dim=1))[:, 0, :] for ins_prompt
                                           in ins_prompts]
                    for i in range(10):
                        ins_prompts[i] = self.bottlenecks_ins_prompts[i](b_ins_prompts_feats[i])
                ins_prompts_cls_scores_list = []
                for i in range(10):
                    score = self.classifiers_ins_prompts[i](ins_prompts[i])
                    # print(ins_prompts[i].mean(dim=1, keepdim=True).size())
                    ins_prompts_cls_scores_list.append(score)
                # ins_prompts_cls_score = sum(ins_prompts_cls_scores_list) / len(ins_prompts_cls_scores_list)
                # ins_prompts_cls_score = [self.classifier_ins_prompts(ins_prompt.reshape(ins_prompt.size(0), ins_prompt.size(1)*ins_prompt.size(2))) for ins_prompt in ins_prompts]
                return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4], ins_prompts_cls_scores_list, dom_logits, dom_proxy_logits, proxy_ids  # global feature for triplet loss
            else:
                return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4], dom_logits, dom_proxy_logits, proxy_ids  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':  # 测试阶段：融合全局和局部特征作为最终输出
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                if cfg.TEST.TSNE:
                    return torch.cat(
                        [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4],
                        dim=1), X_tsne_1, X_tsne_2
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if i not in self.state_dict(): continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model_adr(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type,
                                            rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
