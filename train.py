# -*- coding:utf-8 -*-
from dataset import *
from YOLO_model3 import *
from draw_samples import draw_outputs
from losses import YOloLoss, yolo_boxes, yolo_nms

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os
import cv2

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

FLAGS = easydict.EasyDict({"img_size": 416,

                           "batch_size": 8,

                           "epochs": 50,

                           "num_classes": 20,

                           "lr": 0.001,
                           
                           "tr_img_path": "D:/[1]DB/[3]detection_DB/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
                           
                           "tr_txt_path": "D:/[1]DB/[3]detection_DB/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/xml_to_text",

                           "te_img_path": "D:/[1]DB/[3]detection_DB/voc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
                           
                           "te_txt_path": "D:/[1]DB/[3]detection_DB/voc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/xml_to_text",

                           "class_name": "D:/[1]DB/[3]detection_DB/voc2007/objects-label.txt",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "save_sample": "",
                           
                           "load_weight": "yolov3.tf"})

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def input_func(img_path, lab_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 255.


    return img, lab_path

@tf.function
def cal_loss(model, images, labels, loss):

    with tf.GradientTape() as tape:

        logits = model(images, True)
        regul_loss = tf.reduce_mean(model.losses)
        pred_loss = []
        for output, label, loss_fn in zip(logits, labels, loss):
            pred_loss.append(loss_fn(label, output))
        total_loss = tf.reduce_mean(pred_loss) + regul_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss

def test_sample(model, batch_images, class_name, count):

    for i in range(FLAGS.batch_size):
        images = tf.expand_dims(batch_images[i], 0)
        h = model(images)
        output_0 = h[0]
        output_1 = h[1]
        output_2 = h[2]

        boxes_0 = Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[0]], FLAGS.num_classes),
                            name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[1]], FLAGS.num_classes),
                            name='yolo_boxes_1')(output_1)
        boxes_2 = Lambda(lambda x: yolo_boxes(x, yolo_anchors[yolo_anchor_masks[2]], FLAGS.num_classes),
                            name='yolo_boxes_2')(output_2)

        outputs = Lambda(lambda x: yolo_nms(x, yolo_anchors, yolo_anchor_masks, FLAGS.num_classes),
                            name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
        #
        boxes, scores, classes, nums = outputs
        #
        img = cv2.cvtColor(images[0].numpy() * 255, cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_name)
        cv2.imwrite(FLAGS.save_sample +"/{}_{}.jpg".format(count, i), img)


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
   
    model = YoloV3(FLAGS.img_size, 3, masks=yolo_anchor_masks, classes=FLAGS.num_classes)

    model_pretrained = YoloV3(FLAGS.img_size, 3, masks=yolo_anchor_masks, classes=FLAGS.num_classes)
    model_pretrained.load_weights("C:/Users/Yuhwan/Downloads/ck/yolov3.tf")
    model.get_layer('yolo_darknet').set_weights(
        model_pretrained.get_layer('yolo_darknet').get_weights())
    #freeze_all(model.get_layer('yolo_darknet'))
    
    x_36, x_61, x = model.output
    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(yolo_anchor_masks[0]), FLAGS.num_classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(yolo_anchor_masks[1]), FLAGS.num_classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(yolo_anchor_masks[2]), FLAGS.num_classes, name='yolo_output_2')(x)

    model = tf.keras.Model(model.input, (output_0, output_1, output_2), name='yolov3')
    model.summary()

    class_name = np.loadtxt(FLAGS.class_name, dtype=np.str, skiprows=0, usecols=0)

    tr_txt = os.listdir(FLAGS.tr_txt_path)
    tr_img = [FLAGS.tr_img_path + "/" + data.split('.')[0] + ".jpg" for data in tr_txt]
    tr_lab = [FLAGS.tr_txt_path + "/" + data for data in tr_txt]

    #te_txt = os.listdir(FLAGS.te_txt_path)
    #te_img = [FLAGS.te_img_path + "/" + data.split('.')[0] + ".jpg" for data in te_txt]
    #te_lab = [FLAGS.te_txt_path + "/" + data for data in te_txt]

    #te_generator = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
    #te_generator = te_generator.shuffle(len(te_img))
    #te_generator = te_generator.map(input_func)
    #te_generator = te_generator.batch(1)
    #te_generator = te_generator.prefetch(tf.data.experimental.AUTOTUNE)

    loss = [YOloLoss(yolo_anchors[mask], FLAGS.num_classes) for mask in yolo_anchor_masks]    
    count = 0
    for epoch in range(FLAGS.epochs):

        tr_generator = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
        tr_generator = tr_generator.shuffle(len(tr_img))
        tr_generator = tr_generator.map(input_func)
        tr_generator = tr_generator.batch(FLAGS.batch_size)
        tr_generator = tr_generator.prefetch(tf.data.experimental.AUTOTUNE)

        tr_iter = iter(tr_generator)
        tr_idx = len(tr_img) // FLAGS.batch_size

        for step in range(tr_idx):
            batch_images, batch_labels = next(tr_iter)
            batch_labels = read_label(batch_labels, yolo_anchors, yolo_anchor_masks, FLAGS.img_size, FLAGS.batch_size)
            
            totla_loss = cal_loss(model, batch_images, batch_labels, loss)

            if count % 10 == 0:
                print("YOLO V3 (내 버전): Epoch >> {} [{}/{}]; loss >> {}".format(epoch, step+1, tr_idx, totla_loss))

            if count % 1000 == 0:
                test_sample(model, batch_images, class_name, count)


            count += 1

if __name__ == "__main__":
    main()