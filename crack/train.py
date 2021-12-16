import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from utils.utils import Focal_Loss
from utils.unet_dataset import SegmentationDataset
from nets.runet import Unet

if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    # -------------------------------#
    Cuda = torch.cuda.is_available()
    # -------------------------------#
    #   训练自己的数据集必须要修改的
    #   自己需要的分类个数+1，如2 + 1
    # -------------------------------#
    num_classes = 2
    # --------------------------------------------------------------------------------------------------------------------------
    #   选择主干网络结构：Vgg16 Unet Mobilenet_v3_large  Mobilenet_v3_small
    # ----------------------------------------------------------------------------------------------------------------------------#
    backbone = "Mobilenet_v3_large"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   加载预训练权值，如果设置了model_path，会覆盖pretrained的权值
    # --------------------------------------------------------------------------------------------------------------------------
    pretrained = True
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   加载训练好的权值。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = ""
    # ------------------------------#
    #   输入图片的大小,h,w必须是16的倍数
    # ------------------------------#
    input_shape = [320, 480]
    # ------------------------------#
    #   数据集路径
    # ------------------------------#
    image_dir = "dataset/image"
    mask_dir = "dataset/mask"
    images = "dataset/train.txt"
    # ------------------------------#
    #   模型保存路径
    # ------------------------------#
    save_path = 'weights/road1.pt'
    # ------------------------------#
    #   批处理数据的数量
    # ------------------------------#
    batchsize = 2
    # ------------------------------#
    #   训练的轮数
    # ------------------------------#
    num_epochs = 15
    # ---------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    # ---------------------------------------------------------------------#
    focal_loss = False
    # ---------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ---------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------#
    num_workers = 2
    # ---------------------------------------------------------------------#
    #   初始学习lv
    # ---------------------------------------------------------------------#
    lr = 1e-3
    # ---------------------------------------------------------------------#
    #   学习率衰减策略
    # ---------------------------------------------------------------------#
    lr_decay = False

    model = Unet(pretrained=pretrained, in_channels=3, backbone=backbone).train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()

        pretrained_dict = torch.load(model_path, map_location=device)
        print(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    summary(model.cuda(), (3, 320, 480))
    model.to(device)

    cross_loss = nn.CrossEntropyLoss()
    best_validation_dsc = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.96)

    ds = SegmentationDataset(image_dir, mask_dir, images)
    num_train_samples = ds.num_of_samples()
    dataloader = DataLoader(ds, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True)

    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            images_batch, target_labels = sample_batched['image'], sample_batched['mask']
            if Cuda:
                images_batch, target_labels = images_batch.cuda(), target_labels.cuda()
            optimizer.zero_grad()

            predicted = model(images_batch)
            n, c, h, w = predicted.size()
            nt, ct, ht, wt = target_labels.size()
            if h != ht and w != wt:
                predicted = F.interpolate(predicted, size=(ht, wt), mode="bilinear", align_corners=True)

            target_labels = target_labels.contiguous().view(-1)
            predicted = predicted.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
            target_labels = target_labels.long()

            weights = torch.from_numpy(cls_weights).cuda()
            if focal_loss:
                loss = Focal_Loss(predicted, target_labels, cls_weights=weights, num_classes=num_classes, alpha=0.5,gamma=2)
            else:
                loss =nn.CrossEntropyLoss(weight=weights)(predicted, target_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 计算平均损失
        train_loss = train_loss / num_train_samples

        lr = optimizer.param_groups[0]['lr']
        if lr_decay: lr_scheduler.step()
        print("Epoch: %d Train Loss: %.6f lr: %f" % (epoch + 1, train_loss, lr))


    model.eval()
    torch.save(model, save_path)


