from os.path import join
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from utils.utils import resize_image, cvtColor

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):  
    print('Num classes:', num_classes)
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        #print(gt_imgs[ind][0])
        label = np.array(Image.open(gt_imgs[ind]).convert('L'))
        label[label==255] = 1
        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                    100 * np.nanmean(per_class_iu(hist)),
                                                    100 * np.nanmean(per_class_PA(hist))))
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(round(mPA[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))  
    return mIoUs


def get_miou_png(image, h, w, model, cuda=True):
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image       = cvtColor(image)


    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, (w, h))
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data  = np.expand_dims(np.transpose((np.array(image_data, np.float32))/255.0, (2, 0, 1)), 0)


    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if cuda:
            images = images.cuda()

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        pr = model(images)[0]
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#
        pr = pr[int((h - nh) // 2): int((h - nh) // 2 + nh), int((w - nw) // 2): int((w - nw) // 2 + nw)]
        # ---------------------------------------------------#
        #   进行图片的resize
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = pr.argmax(axis=-1)


    image = Image.fromarray(np.uint8(pr))

    return image