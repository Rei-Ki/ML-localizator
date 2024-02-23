import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def IoU_Loss(true, pred):
    #(32, frames, 4)
    minx1, miny1, maxx1, maxy1 = tf.split(true, 4, axis = 2)
    fminx, fminy, fmaxx, fmaxy = tf.split(pred, 4, axis = 2)
    
    minx2 = tf.minimum(fminx, fmaxx)
    miny2 = tf.minimum(fminy, fmaxy)
    maxx2 = tf.maximum(fminx, fmaxx)
    maxy2 = tf.maximum(fminy, fmaxy)
    
    
    intersection = 0.0 
    
    #найдем пересечение каждого из предсказанных с каждым из реальных
    #сложим все вместе
    for i1 in range(5):
        for i2 in range(5):
            x_overlap = tf.maximum(0.0, tf.minimum(maxx1[:,i1], maxx2[:,i2]) - tf.maximum(minx1[:,i1], minx2[:,i2]))
            y_overlap = tf.maximum(0.0, tf.minimum(maxy1[:,i1], maxy2[:,i2]) - tf.maximum(miny1[:,i1], miny2[:,i2]))
            intersection += x_overlap*y_overlap
            
    #с несколькими обьектами сложнее. Мы не можем просто найди обьединение трех и более прямоугольников по координатам
    #пойдем на некоторые хитрости.
    #не будем считасть обьединение и сравнивать его с пересечение как в IoU
    #а будем стремится сделать площади всех элементов такими-же, как у реальных рамок 
    #просто среднеквадратичной ошибкой
            
    beta1 = 0.0
    for i1 in range(5):
        for i2 in range(5):
            x_overlap = tf.maximum(0.0, tf.minimum(maxx1[:,i1], maxx1[:,i2]) - tf.maximum(minx1[:,i1], minx1[:,i2]))
            y_overlap = tf.maximum(0.0, tf.minimum(maxy1[:,i1], maxy1[:,i2]) - tf.maximum(miny1[:,i1], miny1[:,i2]))
            beta1 += x_overlap*y_overlap
            
    beta2 = 0.0
    for i1 in range(5):
        for i2 in range(5):
            x_overlap = tf.maximum(0.0, tf.minimum(maxx2[:,i1], maxx2[:,i2]) - tf.maximum(minx2[:,i1], minx2[:,i2]))
            y_overlap = tf.maximum(0.0, tf.minimum(maxy2[:,i1], maxy2[:,i2]) - tf.maximum(miny2[:,i1], miny2[:,i2]))
            beta2 += x_overlap*y_overlap
    
    loss = (beta1 - beta2)**2 - intersection
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
