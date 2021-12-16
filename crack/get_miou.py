import os
import torch
from PIL import Image
from tqdm import tqdm

from utils.utils_metrics import compute_mIoU, get_miou_png

'''
预测出来的结果保存在miou_out，因为只有0、1两个值对应["background", "crack"]所以显示为黑色
'''
if __name__ == "__main__":
    # -------------------------------------------------------------------#
    #   验证集损失较低不代表miou较高，本文件用于测试权值在验证集上的泛化性。
    # -------------------------------------------------------------------#
    model_path = 'weights/road.pt'
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   类别名称
    # --------------------------------------------#
    name_classes = ["background", "crack"]
    # -------------------------------------------------------#
    #   数据集所在的文件夹
    # -------------------------------------------------------#
    images = 'dataset/val.txt'
    images_path = 'dataset/image'
    mask_path = 'dataset/mask'

    file = open(images, 'r')
    image_ids = file.read().splitlines()
    file.close()

    pred_dir = "miou_out"

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        unet = torch.load(model_path)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(images_path, image_id + ".jpg")

            image = Image.open(image_path)

            image = get_miou_png(image, 320, 480, unet)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        compute_mIoU(mask_path, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
