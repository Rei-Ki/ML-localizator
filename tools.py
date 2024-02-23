import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# def bbox_to_mask(bboxes, size):
#     """Преобразует bbox в бинарную маску"""
#     masks = []
#     for bbox in bboxes:
#         mask = tf.zeros(size)
#         minx, miny, maxx, maxy = tf.split(bbox, 4, axis = 2)
#         mask[miny:maxy, minx:maxx] = 1
#         masks.append(mask)
#     return tf.stack(masks)

# # Преобразование bbox в маски
# true_masks = bbox_to_mask(t1, size)
# pred_masks = bbox_to_mask(t2, size)

# # Вычисление потерь
# loss = dice_bce_mc_loss(true_masks, pred_masks)


def dice_mc_metric(a, b):
    a = tf.unstack(a, axis=3)
    b = tf.unstack(b, axis=3)
    
    CLASSES = 1
    
    dice_summ = 0
    
    for i, (aa, bb) in enumerate(zip(a, b)):
        numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
        denomerator = tf.math.reduce_sum(aa + bb) + 1
        dice_summ += numenator / denomerator
        
    avg_dice = dice_summ / CLASSES
    
    return avg_dice

def dice_mc_loss(a, b):
    return 1 - dice_mc_metric(a, b)

def dice_bce_mc_loss(a, b):
    return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)


def IoU_Loss(true, pred, frames):
    #(32, frames, 4)
    # print(true.shape)
    # print(pred.shape)
    
    minx1, miny1, maxx1, maxy1 = tf.split(true, 4, axis = 2)
    fminx, fminy, fmaxx, fmaxy = tf.split(pred, 4, axis = 2)
    
    minx2 = tf.minimum(fminx, fmaxx)
    miny2 = tf.minimum(fminy, fmaxy)
    maxx2 = tf.maximum(fminx, fmaxx)
    maxy2 = tf.maximum(fminy, fmaxy)
    
    
    intersection = 0.0 
    
    #найдем пересечение каждого из предсказанных с каждым из реальных
    #сложим все вместе
    for i1 in range(frames):
        for i2 in range(frames):
            x_overlap = tf.maximum(0.0, tf.minimum(maxx1[:,i1], maxx2[:,i2]) - tf.maximum(minx1[:,i1], minx2[:,i2]))
            y_overlap = tf.maximum(0.0, tf.minimum(maxy1[:,i1], maxy2[:,i2]) - tf.maximum(miny1[:,i1], miny2[:,i2]))
            intersection += x_overlap*y_overlap
            
    #с несколькими обьектами сложнее. Мы не можем просто найди обьединение трех и более прямоугольников по координатам
    #пойдем на некоторые хитрости.
    #не будем считасть обьединение и сравнивать его с пересечение как в IoU
    #а будем стремится сделать площади всех элементов такими-же, как у реальных рамок 
    #просто среднеквадратичной ошибкой
            
    beta1 = 0.0
    for i1 in range(frames):
        for i2 in range(frames):
            x_overlap = tf.maximum(0.0, tf.minimum(maxx1[:,i1], maxx1[:,i2]) - tf.maximum(minx1[:,i1], minx1[:,i2]))
            y_overlap = tf.maximum(0.0, tf.minimum(maxy1[:,i1], maxy1[:,i2]) - tf.maximum(miny1[:,i1], miny1[:,i2]))
            beta1 += x_overlap*y_overlap
            
    beta2 = 0.0
    for i1 in range(frames):
        for i2 in range(frames):
            x_overlap = tf.maximum(0.0, tf.minimum(maxx2[:,i1], maxx2[:,i2]) - tf.maximum(minx2[:,i1], minx2[:,i2]))
            y_overlap = tf.maximum(0.0, tf.minimum(maxy2[:,i1], maxy2[:,i2]) - tf.maximum(miny2[:,i1], miny2[:,i2]))
            beta2 += x_overlap*y_overlap
    
    loss = (beta1 - beta2)**2 - intersection
    return loss


def IoU_Loss_one(true, pred):
    #(32, 1, 4) 
    t1 = true
    t2 = pred
    
    ious = []
    for i in range(t1.shape[0]):
        mask = true[i]
        pred_tmp = pred[i]
        masks_sum = tf.reduce_sum(mask)
        predictions_sum = tf.reduce_mean(pred_tmp)
        intersection = tf.reduce_sum(tf.multiply(mask, pred_tmp))
        union = masks_sum + predictions_sum - intersection
        iou = intersection / union
        loss = 1 - iou
        ious.append(loss)
    return ious
    
    #наши данные уже в правильном порядке
    minx1, miny1, maxx1, maxy1 = tf.split(t1, 4, axis = 2)
    
    #minx1.shape = (32,1,1) работаем сразу с целым батчем
    
    #а вот нейросеть не знает, где должна быть min и max координата
    #нормализуем данные
    fminx, fminy, fmaxx, fmaxy = tf.split(t2, 4, axis = 2)
    minx2 = tf.minimum(fminx, fmaxx)
    miny2 = tf.minimum(fminy, fmaxy)
    maxx2 = tf.maximum(fminx, fmaxx)
    maxy2 = tf.maximum(fminy, fmaxy)
    
    #считаем пересечение прямо как в алгоритме выше
    x_overlap = tf.maximum(0.0, tf.minimum(maxx1, maxx2) - tf.maximum(minx1, minx2))
    y_overlap = tf.maximum(0.0, tf.minimum(maxy1, maxy2) - tf.maximum(miny1, miny2))
    
    intersection = x_overlap*y_overlap
    
    #площади найти не сложно
    area1 = (maxx1 -minx1)*(maxy1-miny1)
    area2 = (maxx2 -minx2)*(maxy2-miny2)
    
    #обьединение тоже по алгоритму
    union = area1 + area2 - intersection
    
    IoU = intersection/union
    loss = 1.0 - IoU
    return loss


def check_image_frame(dataset):
    for i, c in dataset.take(1):
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(3, 1, 1)
        i = i.numpy()
        c = c.numpy()
        c = c.astype(np.int16)
        for bb in c:
            print(bb)
            i = cv2.rectangle(i ,bb[0],bb[1],(0,1,0),1)
        plt.imshow(i)
        plt.show()
