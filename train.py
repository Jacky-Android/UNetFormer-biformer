from torch.utils.data import DataLoader
from potsdam_data import *
from models.UNetFormer import UNetFormer
from train_utils import train_one_epoch, evaluate, create_lr_scheduler

def create_model(num_classes):
    model = UNetFormer(
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='resnet18',
                 pretrained=False,
                 window_size=8,
                 num_classes=num_classes)
    return model

train_batch_size = 1
val_batch_size = 1
batch_size = 1

train_dataset = PotsdamDataset(data_root='G:/postam_orignal/kaggle/working', mode='train',img_dir='images/test', mask_dir='anns/test',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(transform=val_aug)
test_dataset = PotsdamDataset(data_root='G:/postam_orignal/kaggle/working',img_dir='images/test', mask_dir='anns/test',
                              transform=val_aug)
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=num_workers,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=6)
model.to(device)
params_to_optimize = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=0.01, momentum=0.9, weight_decay=1e-4,
    )

amp = False
scaler = torch.cuda.amp.GradScaler() if amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 10, warmup=True)

num_classes = 6
best_dice = 0

for epoch in range(0,10):
        
            
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=100, scaler=scaler)
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        