import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

import sys
sys.path.append('.')
# from model.ResNet import create_model
# from model.resnet import create_model
from model.Res2Net import create_model


from torchvision.models import resnet18
# seed_everything(7)
# AVAIL_GPUS = min(1, torch.cuda.device_count())
# BATCH_SIZE = 256 if AVAIL_GPUS else 64
# NUM_WORKERS = int(os.cpu_count() / 2)
# max_steps = 60 * 10e4
# n_crop = 10
IMAGENET_NUM_CLASSES = 200

class LitResNet(LightningModule):
    def __init__(self, version, num_classes, plain, option, initial_lr=0.1):
        super().__init__()

        # 将超参数设置为属性
        # self.save_hyperparameters()
        self.version = version
        self.num_classes = num_classes
        self.plain = plain
        self.initial_lr = initial_lr
        self.option = option
        
        # 创建模型
        self.save_hyperparameters()
        # self.model = create_model(version, num_classes, plain)
        self.model = create_model(version, num_classes=self.num_classes, plain=self.plain, option=option)
    
    def to_fully_conv(self):
        # 将fc层替换为卷积
        print('替换为全卷积网络')
        fc_conv = nn.Conv2d(in_channels=self.model.fc.in_features, out_channels=self.num_classes, kernel_size=1)
        fc_conv.weight.data.copy_(self.model.fc.weight.data.view(*self.model.fc.weight.data.shape, 1, 1))
        fc_conv.bias.data.copy_(self.model.fc.bias.data)
        self.model.fc = fc_conv
        self.model.fully_conv = True

    def forward(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        prob = F.softmax(logits, dim=1)
        # log_prob = torch.log(prob)
        # loss = F.nll_loss(log_prob, y)
        loss = F.cross_entropy(logits, y)
        # logits = F.log_softmax(self.model(x), dim=1)
        # loss = F.nll_loss(logits, y)
        top1_err = 1 - accuracy(prob, y, top_k=1)
        top5_err = 1 - accuracy(prob, y, top_k=5)

        self.log('train_loss', loss)
        self.log('train_top1_err', top1_err, prog_bar=True)
        self.log('train_top5_err', top5_err, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y =batch
        if self.num_classes == IMAGENET_NUM_CLASSES and stage == 'test':
            bs, ncrops, c, h, w = x.size()
            x = x.view(-1, c, h, w) # fuse batch size and ncrops
        logits = self(x)
        if self.num_classes == IMAGENET_NUM_CLASSES and stage == 'test':
            logits = logits.view(bs, ncrops, -1).mean(1)
        
        prob = F.softmax(logits, dim=1)
        log_prob = torch.log(prob)
        loss = F.nll_loss(log_prob, y) # 等同于crossentropy loss
        # preds = torch.argmax(logits, dim=1)
        top1_err = 1 - accuracy(prob, y, top_k=1)
        top5_err = 1 - accuracy(prob, y, top_k=5)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_top1_err', top1_err, prog_bar=True)
            self.log(f'{stage}_top5_err', top5_err, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_lr, momentum=0.9, weight_decay=0.0001)
        
        if self.num_classes == IMAGENET_NUM_CLASSES:
            # ImageNet 的 lr 变化策略
            scheduler_dict = {
                'scheduler': ReduceLROnPlateau(optimizer,mode='min', factor=0.1),
                'monitor': 'val_loss',
                'strict': False,
            }
        else:
            scheduler_dict = {
                'scheduler': MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1),
                'strict': False,
                'interval': 'step'
            }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

