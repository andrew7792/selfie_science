# import dlib as dlib
import numpy as np
from PIL import Image
import tiny_face_model
import tensorflow as tf
import pylab as pl
import cv2
import imutils
import json
from pprint import pprint
from age_gender import ageGender
from util import nms
from keras.models import load_model
from scipy.special import expit
from imutils.face_utils import FaceAligner
from keras.backend.tensorflow_backend import set_session

weight_file_path = 'model_data/mat2tf.pkl'
prob_thresh = 0.98
nms_thresh = 0.1

MAX_INPUT_DIM = 5000.0

# config = tf.ConfigProto()
config = tf.ConfigProto(device_count={'GPU': 0})

model_face = tiny_face_model.Model(weight_file_path)

sess = tf.Session(config=config)
set_session(sess)

embedding_model = load_model('model_data/facenet_keras.h5')
embedding_model.load_weights('model_data/facenet_keras_weights.h5')

x = tf.placeholder(tf.float32, [1, None, None, 3])
score_final = model_face.tiny_face(x)

average_image = model_face.get_data_by_key("average_image")
clusters = model_face.get_data_by_key("clusters")

# predictor = dlib.shape_predictor('landmarks_detector/shape_predictor_68_face_landmarks.dat')

# fa = FaceAligner(predictor, desiredFaceWidth=256)

sess.run(tf.global_variables_initializer())


def overlay_bounding_boxes(raw_img, refined_bboxes):
    if len(refined_bboxes):
        for r in refined_bboxes:
            _r = [int(x) for x in r[:4]]
            bounding_box = _r
        return (raw_img[_r[1]:_r[3], _r[0]:_r[2]], bounding_box)


def calc_scales(model, raw_img, clusters):
    normal_idx = np.where(clusters[:, 4] == 1)
    clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
    min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                    np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
    max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
    scales_down = pl.arange(min_scale, 0, 1.)
    scales_up = pl.arange(0.5, max_scale, 0.5)
    scales_pow = np.hstack((scales_down, scales_up))
    scales = np.power(2.0, scales_pow)
    return scales


def calc_bounding_boxes(prob_cls_tf, score_reg_tf, score_cls_tf, s):
    # threshold for detection
    _, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)
    cy = fy * 8 - 1
    cx = fx * 8 - 1
    ch = clusters[fc, 3] - clusters[fc, 1] + 1
    cw = clusters[fc, 2] - clusters[fc, 0] + 1

    # extract bounding box refinement
    Nt = clusters.shape[0]
    tx = score_reg_tf[0, :, :, 0:Nt]
    ty = score_reg_tf[0, :, :, Nt:2 * Nt]
    tw = score_reg_tf[0, :, :, 2 * Nt:3 * Nt]
    th = score_reg_tf[0, :, :, 3 * Nt:4 * Nt]

    # refine bounding boxes
    dcx = cw * tx[fy, fx, fc]
    dcy = ch * ty[fy, fx, fc]
    rcx = cx + dcx
    rcy = cy + dcy
    rcw = cw * np.exp(tw[fy, fx, fc])
    rch = ch * np.exp(th[fy, fx, fc])

    scores = score_cls_tf[0, fy, fx, fc]

    tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
    tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
    tmp_bboxes = tmp_bboxes.transpose()
    return tmp_bboxes


def getFace(image):
    raw_img_bgr = np.asarray(image)
    raw_img = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
    raw_img_f = raw_img.astype(np.float32)

    scales = calc_scales(model_face, raw_img, clusters)

    bboxes = np.empty(shape=(0, 5))  # initialize output

    for s in scales:  # process input at different scales
        img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        img = img - average_image
        img = img[np.newaxis, :]

        # we don't run every template on every scale ids of templates to ignore
        tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
        ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

        # run through the net
        score_final_tf = sess.run(score_final, feed_dict={x: img})

        # collect scores
        score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
        prob_cls_tf = expit(score_cls_tf)
        prob_cls_tf[0, :, :, ignoredTids] = 0.0

        tmp_bboxes = calc_bounding_boxes(prob_cls_tf, score_reg_tf, score_cls_tf, s)
        bboxes = np.vstack((bboxes, tmp_bboxes))

    refind_idx = nms(bboxes, nms_thresh)
    refined_bboxes = bboxes[refind_idx]

    # convert PIL Image to OpenCV Image
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    origin_img = image_cv.copy()

    if len(bboxes) == 0:
        return False
    for refined_bbox in refined_bboxes:
        bbox = refined_bbox.astype(np.int64)

        cv2.rectangle(origin_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        orig_w = image_cv.shape[0]
        orig_h = image_cv.shape[1]

        face_h = int((bbox[3] - bbox[1])*1.5)
        face_w = int((bbox[2] - bbox[0])*1.5)
        face_x = bbox[0] - int(face_w/2)
        face_y = bbox[1] - int(face_h/2)

        if face_x < 0:
            face_x = 0
        if face_y < 0:
            face_y = 0
        if face_w > orig_w:
            face_w = orig_w - 2
        if face_h > orig_h:
            face_h = orig_h - 2

        crop_face = image_cv[face_y:face_y + face_h, face_x:face_x + face_w].copy()

        age, gender, face_cv2 = ageGender(crop_face)
        print(age, gender)

        if age is not False:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (bbox[0], bbox[1]-15)
            fontScale = 1
            fontColor = (0, 0, 255)
            lineType = 2

            cv2.putText(origin_img, age+gender,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)


        # cv2.rectangle(image_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    cv2.imwrite('output.jpg', origin_img)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow('output', origin_img)
    cv2.resizeWindow('output', 600, 600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(refined_bboxes)


image_path = 'images/1660315892310801029.jpg'
# image_path = 'images/2189357871670292920.jpg'
im = Image.open(image_path)

faces = getFace(im)
print(faces)

