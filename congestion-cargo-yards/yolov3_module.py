import numpy as np
import cv2

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.

def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, is_proportional):
    if is_proportional:
        scale = np.array([min(im_w/im_h, 1), min(im_h/im_w, 1)])
        offset = 0.5*(np.ones(2) - scale)
        x, y = (np.array([x, y]) - offset) / scale
        width, height = np.array([width, height]) / scale
    xmin = int((x - width / 2) * im_w)
    ymin = int((y - height / 2) * im_h)
    xmax = int(xmin + width * im_w)
    ymax = int(ymin + height * im_h)
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())

def parse_yolo_region(predictions, resized_image_shape, original_im_shape, params, threshold, is_proportional):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = predictions.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    size_normalizer = (resized_image_w, resized_image_h) if params.isYoloV3 else (params.side, params.side)
    bbox_size = params.coords + 1 + params.classes
    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for row, col, n in np.ndindex(params.side, params.side, params.num):
        # Getting raw values for each detection bounding box
        bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < threshold:
            continue
        # Process raw value
        x = (col + x) / params.side
        y = (row + y) / params.side
        # Value for exp is very big number in some cases so following construction is using here
        try:
            width = np.exp(width)
            height = np.exp(height)
        except OverflowError:
            continue
        # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
        width = width * params.anchors[2 * n] / size_normalizer[0]
        height = height * params.anchors[2 * n + 1] / size_normalizer[1]

        class_id = np.argmax(class_probabilities)
        confidence = class_probabilities[class_id]*object_probability
        if confidence < threshold:
            continue
        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, is_proportional=is_proportional))
    return objects

def get_objects(output, net, new_frame_height_width, source_height_width, prob_threshold, is_proportional):
    objects = list()

    for layer_name, out_blob in output.items():
        out_blob = out_blob.buffer.reshape(net.layers[net.layers[layer_name].parents[0]].out_data[0].shape)
        layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
        objects += parse_yolo_region(out_blob, new_frame_height_width, source_height_width, layer_params,
                                     prob_threshold, is_proportional)

    return objects

def filter_objects(objects, iou_threshold, prob_threshold):
    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    return tuple(obj for obj in objects if obj['confidence'] >= prob_threshold)

# IoU calculation
def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union, area_of_overlap, area_of_union

"""
https://github.com/nscalo/OpenVINO-YOLOV4/blob/master/object_detection_demo_yolov3_async.py
"""
def draw_yolo_bounding_boxes(outputs, net, input_width, input_height, frame, args):
    objects = get_objects(outputs, net, (input_height, input_width), frame.shape[:-1], args.prob_threshold,
                            args.keep_aspect_ratio)
    objects = filter_objects(objects, args.iou_threshold, args.prob_threshold)
    origin_im_size = frame.shape[:-1]
    confs = []
    for obj in objects:
        # Validation bbox of detected object
        obj['xmax'] = min(obj['xmax'], origin_im_size[1])
        obj['ymax'] = min(obj['ymax'], origin_im_size[0])
        obj['xmin'] = max(obj['xmin'], 0)
        obj['ymin'] = max(obj['ymin'], 0)
        color = (min(obj['class_id'] * 12.5, 255),
                    min(obj['class_id'] * 7, 255),
                    min(obj['class_id'] * 5, 255))
        # det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
        #     str(obj['class_id'])

        confs.append(obj['confidence'])
        cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        # cv2.putText(frame,
        #             "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
        #             (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    return frame, confs