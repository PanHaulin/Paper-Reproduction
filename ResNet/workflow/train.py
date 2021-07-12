from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torchvision
import torch
import os
import sys
import argparse
from pprint import PrettyPrinter
from torchvision import transforms
from torchvision import version
from torchvision.transforms import ToTensor, Lambda, Normalize
from torchvision.transforms.transforms import RandomCrop, RandomHorizontalFlip, Resize
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from thop import profile, clever_format
from PIL import Image
sys.path.append('.')

from model.LitResNet import LitResNet
from utils.preprocessing import ShortEdgeScale, MeanSubstractAndColorShift, MultiScaleCrops
from utils.datamodule import TinyImagenetDataModule

seed_everything(7)
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)

# logger online
API_KEY = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMWRkNzhhZi04NTYzLTRkNTktYWM1Yi1iODdlZWI2NTNmMTMifQ=='
PROJECT_NAME = 'porient/resnet-exp'

def train(args):
    # set params
    dataset = args.dataset
    fast_dev_run = args.fast_dev_run
    offline = args.offline
    plain = args.plain
    version = args.version
    option = args.option

    # 数据预处理
    if dataset.lower() == 'imagenet':
        print('train ResNet{} with plain={} on tiny_imagenet'.format(version, plain))
        # tiny-imagenet 64*64 但会resize到224*224
        # imagenet2012 num_classes=1000, train_per_class=1000, val_per_class=50, test_per_class=100
        # tiny-imagenet num_classes=200, train_per_class=450, val_per_class=50, test_per_class=50, 训练量缩小了约10倍
        num_classes = 200 #1000
        max_steps = 60e4 #tiny 对应的约为3*10e4
        BATCH_SIZE = 256 if AVAIL_GPUS else 64
        steps_per_epoch = 1281167 / BATCH_SIZE # 5000
        # max_epochs = int(max_steps / steps_per_epoch) # 约120epochs
        max_epochs = 120
        # max_epochs = 3
        # input_size = (1, 3, 224, 224)
        # input = torch.randn(1, 3, 224, 224).cuda()
        input = torch.randn(1, 3, 224, 224)

        # normalize = torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)) #Tiny-ImageNet
        # normalize = torchvision.transforms.Normalize((0.485,0.456,0.406), (1,1,1)) #ImageNet
        DATASET_PATH = 'datasets/tiny-imagenet-200/'
        # train_transforms = torchvision.transforms.Compose([
        #     ShortEdgeScale(size_min=256, size_max=480),
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.RandomCrop(size=224),
        #     # MeanSubstractAndColorShift(DATASET_PATH + '/train/'),
        #     torchvision.transforms.ToTensor(),
        #     normalize,  # 论文中只提到减去均值(0.485,0.456,0.406), (0.229,0.224,0.225)
        #     # torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        # ])
        
        # test_transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(224),
        #     MultiScaleCrops(size_list=[224,256,384,480,640]),
        #     Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        # ])
        # train_transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(224, interpolation=Image.LANCZOS),
        #     ShortEdgeScale(size_min=256, size_max=480),
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.RandomCrop(size=224),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        # ])
        
        # test_transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(224, interpolation=Image.LANCZOS),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        # ])
        train_transforms = torchvision.transforms.Compose([
            # ShortEdgeScale(size_min=256, size_max=480),
            ShortEdgeScale(size_min=72, size_max=136),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (1, 1, 1)),

        ])
        
        test_transforms = torchvision.transforms.Compose([
            # MultiScaleCrops(size_list=[224,256,384,480,640]),
            MultiScaleCrops(size_list=[64,72,108,136,182]),
            Lambda(lambda crops: torch.stack([Normalize((0.4802, 0.4481, 0.3975), (1, 1, 1))(ToTensor()(crop)) for crop in crops])),
        ])
        
        dm = TinyImagenetDataModule(
            data_dir=DATASET_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_transforms=train_transforms,
            val_transforms=test_transforms,
            test_transforms=test_transforms
        )

 
    elif dataset.lower() == 'cifar10':
        print('train ResNet{} with plain={} on cifar10'.format(version, plain))
        num_classes = 10
        max_steps = 64000
        BATCH_SIZE = 128 if AVAIL_GPUS else 64
        steps_per_epoch = 45000 / BATCH_SIZE
        max_epochs = int(max_steps / steps_per_epoch) #约180epochs
        # max_epochs = 50
        # input_size = (1, 3, 32, 32)
        input = torch.randn(1, 3, 32, 32).cuda()
        
        DATASET_PATH = 'datasets/cifar10/'
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Pad(padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dm = CIFAR10DataModule(
            data_dir=DATASET_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            val_split=0.1
        )
        dm.train_transforms = train_transforms
        dm.test_transforms = test_transforms
        dm.val_transforms  = test_transforms

    model = LitResNet(version, num_classes, plain, option=option, initial_lr=0.1)

    # logger
    if plain:
        type='plain'
    else:
        type='residual'
    log_params = {
        'dataset': dataset,
        'version': version,
        'batch_size': BATCH_SIZE,
        'option': option,
        'max_epochs': max_epochs,
        'type': type,
    }

    logger = NeptuneLogger(
        offline_mode = offline,
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        params=log_params,
        close_after_fit=False
    )

    # callbacks
    dirpath='checkpoints/'+str(version)

    if plain:
        filename=dataset+'-plain-{epoch:02d}-{val_loss:.2f}'
    else:
        filename=dataset+'-residual-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=dirpath,
        filename=filename,
        save_top_k=1,
        mode='min',
    )

    # trainer
    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=max_epochs,
        gpus=1,
        auto_select_gpus=True,
        logger=logger,
        fast_dev_run=fast_dev_run,
        callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback],
    )

    trainer.fit(model, dm)

    # load best and test
    if not fast_dev_run:
        best_path = checkpoint_callback.best_model_path
        print('best_path:{}'.format(best_path))
        model = model.load_from_checkpoint(best_path)
    model.to_fully_conv()
    trainer.test(model, datamodule=dm)

    # log statistic
    # statistic info
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    logger.log_hyperparams({
        'flops': flops,
        'params': params
        })

    #stop exp
    logger.experiment.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="[str] a dataset name declared in config file", default=None, type=str)
    parser.add_argument("--version", help="[str] resnet version", default=None, type=int)
    parser.add_argument('--fast_dev_run',help="when declare, runs 1 train, val, test batch and program ends", action='store_true', default=False)
    parser.add_argument('--offline', help='logger online', action='store_true', default=False)
    parser.add_argument('--plain', help='train without shortcut', action='store_true', default=False)
    parser.add_argument("--option", help="[str] a dataset name declared in config file", default='B', type=str)
    args = parser.parse_args()
    pp = PrettyPrinter(indent=2)
    print('args:')
    pp.pprint(args)

    train(args)
    
if __name__ == '__main__':
    main()
