"""Congestion on cargo yards."""


import os
import sys
import time
import socket
import json
import numpy as np
import cv2

from argparse import ArgumentParser
from inference import Network
from concurrent.futures import ThreadPoolExecutor

import logging
import threading

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    model_desc = """
    Path to an xml file with a trained model.
    """
    video_desc = """
    Path to image or video file
    """
    cpu_extension_desc = """
    MKLDNN (CPU)-targeted custom layers.
    Absolute path to a shared library with the
    kernels impl.
    """
    device_desc = """
    Specify the target device to infer on: 
    CPU, GPU, FPGA or MYRIAD is acceptable. Sample 
    will look for a suitable plugin for device 
    specified (CPU by default)
    """
    prob_threshold_desc = """
    Probability threshold for detections filtering
    (0.5 by default)
    """
    color_desc = """
    color for painting
    """
    thickness_desc = """
    thickness for painting
    """
    batch_size_desc = """
    batch_size for inference
    """
    mode_desc = """
    mode for async, sync
    """

    parser = ArgumentParser()
    parser.add_argument("-m", help=model_desc, required=True, type=str)
    parser.add_argument("-i", help=video_desc, required=False, type=str)
    parser.add_argument("-vds", help=video_desc, required=False, type=str)
    parser.add_argument("-l", help=cpu_extension_desc, required=False, type=str, default=None)
    parser.add_argument("-d", help=device_desc, type=str, default="CPU")
    parser.add_argument('-xp', required=False, type=str)
    parser.add_argument("-pt", help=prob_threshold_desc, type=float, default=0.5)
    parser.add_argument("-c", help=color_desc, type=tuple, default=(0,255,0))
    parser.add_argument("-th", help=thickness_desc, type=int, default=2)
    parser.add_argument("-bt", "--batch_size", help=batch_size_desc, type=int, default=16)
    parser.add_argument("-mode", "--mode", help=mode_desc, type=str, default='async')
    parser.add_argument("-out", "--output_log", help=mode_desc, type=str, default='main.log')
    parser.add_argument("-o", "--output_video", help=mode_desc, type=str, default="output_video.avi")
    parser.add_argument("-thr", "--threshold", help=mode_desc, type=float, default=0.4)
    parser.add_argument("-ic", "--is_conf", help="", type=bool, default=False)
    parser.add_argument("-asp", "--aspect", help="", type=bool, default=False)
    parser.add_argument("-it", "--is_threshold", help="", type=bool, default=False)
    parser.add_argument("-gs", "--grayscale", help="", type=bool, default=False)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
            "detections filtering", default=0.4, type=float)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
            default=0.5, type=float)
    
    return parser

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
            width = exp(width)
            height = exp(height)
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

def preprocessing(frame, net_input_shape, grayscale=False):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    if grayscale:
        p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        p_frame = np.expand_dims(p_frame, 2)
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)

    return p_frame

def calculate_threshold(frame, result, args, width, height):
    r = (result.flatten() > args.threshold).astype(bool)
    return np.sum(r)

"""
https://github.com/nscalo/OpenVINO-YOLOV4/blob/master/object_detection_demo_yolov3_async.py
"""
def draw_yolo_bounding_boxes(outputs, net, input_width, input_height, frame, args):
    objects = get_objects(outputs, net, (input_height, input_width), frame.shape[:-1], args.prob_threshold,
                            args.keep_aspect_ratio)
    objects = filter_objects(objects, args.iou_threshold, args.prob_threshold)
    origin_im_size = frame.shape[:-1]
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

        cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        # cv2.putText(frame,
        #             "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
        #             (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

def draw_boxes(frame, result, args, width, height):
    confs = []
    result = result.reshape(1, 425, 13, 13)
    for i in range(13):
        for j in range(13):
            boxes = result[0,:,i,j]: # Output shape is 1x1x100x7
            boxes = boxes.reshape(-1,85)
            for box in boxes:
                conf = box[4]
                if conf >= args.pt:
                    xmin = int(box[0] * width)
                    ymin = int(box[1] * height)
                    xmax = int(box[0] * width + box[3] * width)
                    ymax = int(box[1] * height + box[2] * width)
                    confs.append(conf)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, args.th)
            
    return frame, confs

def infer_on_batch_result(infer_network, frames, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    results = infer_network.extract_output(request_id=request_id)
    output_shape = 200
    thrs = []
    confidences = []

    if aspect:
        return [], [], []

    ### TODO: Update the frame to include detected bounding boxes
    for ii, frame in enumerate(frames):
        if threshold:
            thr = calculate_threshold(frame, results[ii,:,:,:], args, width, height)
            thrs.append(thr)
        else:
            frame, confs = draw_boxes(frame, results[ii,:,:,:], args, width, height)
        frames[ii] = frame
        confidences = np.append(confidences, confs).tolist()
    
    # Write out the frame
    
    return frames, thrs, confidences

def infer_on_multi_result(infer_network, frames, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    results = infer_network.extract_output(request_id=request_id)
    output_shape = 200
    thrs = []
    confidences = []

    if aspect:
        return [], [], []

    ### TODO: Update the frame to include detected bounding boxes
    for ii, frame in enumerate(frames):
        if threshold:
            thr = calculate_threshold(frame, results[:,:,output_shape*ii:output_shape*ii+output_shape,:], 
            args, width, height)
            thrs.append(thr)
        else:
            frame, confs = draw_boxes(frame, results[:,:,output_shape*ii:output_shape*ii+output_shape,:], 
        args, width, height)
        frames[ii] = frame
        confidences = np.append(confidences, confs).tolist()
    
    # Write out the frame
    
    return frames, thrs, confidences

def infer_on_result(infer_network, frame, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    results = infer_network.extract_output(request_id=request_id)
    output_shape = 200
    thrs = []
    confidences = []

    if aspect:
        return [], [], []

    ### TODO: Update the frame to include detected bounding boxes
    if threshold:
        thr = calculate_threshold(frame, results[:,:,:,:], 
        args, width, height)
        thrs.append(thr)
    else:
        frame, confs = draw_boxes(frame, results, args, width, height)
        confidences = np.append(confidences, confs).tolist()
    
    return frame, thrs, confidences

def sync_async(infer_network, p_frames, t='sync', request_id=20):
    if t == "async":
        for ii, p_frame in enumerate(p_frames[0:16]):
            infer_network.async_inference(p_frame, request_id=ii+request_id)
    elif t == "sync":
        infer_network.sync_inference(p_frames[16:])

def infer_on_stream(args, client=None):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    videos = args.vds.split(",")
    caps = []
    widths = []
    heights = []
    outs = []

    for video in videos:
        c = cv2.VideoCapture(video)
        c.open(video)
        widths.append(int(c.get(3)))
        heights.append(int(c.get(4)))
        caps.append(c)

    def call_each_video_capture(args, cap, num_index, width, height, output_video, out_data_file_name):

        # Initialise the class
        infer_network = Network()

        CPU_EXTENSION = args.l

        def exec_f(l):
            pass

        infer_network.load_core(args.m, args.d, cpu_extension=CPU_EXTENSION, args=args)

        if "MYRIAD" in args.d:
            infer_network.feed_custom_layers(args, {'xml_path': args.xp}, exec_f)

        if "CPU" in args.d:
            infer_network.feed_custom_parameters(args, exec_f)

        infer_network.load_model(args.m, args.d, cpu_extension=CPU_EXTENSION, args=args)
        
        # Set Probability threshold for detections

        args = build_argparser().parse_args()

        ### TODO: Load the model through `infer_network` ###
        infer_network.load_model(args.m, args.d, cpu_extension=args.l, 
        args=args)

        frames = []
        
        # if (num_index != 1):
        #     out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width,height))

        def call_infer_network(infer_network, frame, args, width, height, request_id, conf=False, threshold=False, aspect=False):
            if infer_network.wait(request_id=request_id) == 0:
                frame, thrs, confidences = infer_on_result(infer_network, frame, args, width, height, request_id, 
                conf=conf, threshold=threshold, aspect=args.aspect)
            return frame, thrs, confidences

        def call_sync_infer_network(infer_network, frames, args, width, height, request_id, conf=False, threshold=False, aspect=False):
            frames, thrs, confidences = infer_on_multi_result(infer_network, frames, args, width, height, request_id, 
            conf=conf, threshold=threshold, aspect=args.aspect)
            return frames, thrs, confidences

        def call_sync_batch_infer_network(infer_network, frames, args, width, height, request_id, conf=False, threshold=False, aspect=False):
            frames, thrs, confidences = infer_on_batch_result(infer_network, frames, args, width, height, request_id, 
            conf=conf, threshold=threshold, aspect=args.aspect)
            return frames, thrs, confidences

        def call_sync_async_infer_network(infer_network, frames, args, width, height, request_id, conf=False, threshold=False, aspect=False):
            for ii, frame in enumerate(frames[0:16]):
                frame, thrs1, confidences1 = call_infer_network(infer_network, frame, args, width, height, 
                request_id=ii+request_id, conf=conf, threshold=threshold, aspect=args.aspect)
                frames[ii] = frame
            frames[16:], thrs2, confidences2 = call_sync_infer_network(infer_network, frames[16:], args, width, height, 
            request_id=request_id, conf=conf, threshold=threshold, aspect=args.aspect)

            return frames, np.append(thrs1, thrs2).tolist(), np.append(confidences1, confidences2).tolist()

        counter = 1
        ix = []
        cx = []

        ### TODO: Loop until stream is over ###
        while cap.isOpened():
            
            ### TODO: Read from the video capture ###
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

            if (counter) % args.batch_size == 0:

                start_time = time.time()
                
                ### TODO: Pre-process the image as needed ###
                p_frames = []

                for frame in frames:
                    p_frame = preprocessing(frame, infer_network.get_input_shape(), args.grayscale)
                    p_frames.append(p_frame)
                
                end_time = time.time()
                logging.info("""Frame preprocessing time: {t}""".format(t=(end_time - start_time)))

                start_time = time.time()
                
                ### TODO: Start asynchronous inference for specified request ###
                if args.mode == "async":
                    for ii, p_frame in enumerate(p_frames):
                        infer_network.async_inference(p_frame, request_id=ii)
                    end_time = time.time()
                elif args.mode == "sync" or args.mode == "sync_batch":
                    infer_network.sync_inference(p_frames)
                    end_time = time.time()
                elif args.mode == "sync_async":
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        for ii in executor.map(sync_async, [infer_network]*2, [p_frames]*2, 
                        ['sync', 'async'], [5]*2):
                            pass
                    end_time = time.time()
                
                logging.info("""Frame inference time: {t}""".format(t=(end_time - start_time)))

                ### TODO: Wait for the result ###
                start_time = time.time()
                if args.mode == "async":
                    for ii, frame in enumerate(frames):
                        frame, thrs, confidences = call_infer_network(infer_network, 
                        frame, args, width, height, ii, conf=args.is_conf, 
                        threshold=args.is_threshold, aspect=args.aspect)
                        frames[ii] = frame
                elif args.mode == "sync":
                    frames, thrs, confidences = call_sync_infer_network(infer_network, frames, 
                    args, width, height, 0, conf=args.is_conf, 
                    threshold=args.is_threshold, aspect=args.aspect)
                elif args.mode == "sync_batch":
                    frames, thrs, confidences = call_sync_batch_infer_network(infer_network, frames, 
                    args, width, height, 0, conf=args.is_conf, 
                    threshold=args.is_threshold, aspect=args.aspect)
                elif args.mode == "sync_async":
                    frames, thrs, confidences = call_sync_async_infer_network(infer_network, frames, 
                    args, width, height, 5, conf=args.is_conf, 
                    threshold=args.is_threshold, aspect=args.aspect)

                end_time = time.time()
                
                logging.info("""Thresholds count: {t}""".format(t=(np.mean(thrs))))
                logging.info("""Frame extract time: {t}""".format(t=(end_time - start_time)))

                start_time = time.time()
                
                if ("png" in args.vds or "jpg" in args.vds):
                    for frame in frames:
                        cv2.imwrite(output_video, frame)
                if ("mp4" in args.vds or "avi" in args.vds):
                    for frame in frames:
                        out.write(frame)

                end_time = time.time()

                logging.info("""Frame paint time: {t}""".format(t=(end_time - start_time)))

                frames = []

            counter += 1
            
        # if args.output_video:
        #     out.release()
        
        cap.release()

    threads = []
    for num_index, cap in enumerate(caps):
        t = threading.Thread(target=call_each_video_capture, 
        args=(args, cap, num_index, widths[num_index], heights[num_index], 
        str(videos[num_index][::-1][3:][::-1]) + str(num_index) + ".png", str(videos[num_index][::-1][3:][::-1]) + str(num_index) + "_.npy"))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    cv2.destroyAllWindows()
    
    return None

def convert_perf_time(perf_text):

    import re
    
    result = re.split("\n", perf_text)

    order = ['preprocessing', 'inference', 'extract', 'paint']

    def extract(r):
        l = r.split(" ")
        return l[len(l)-1]

    result = list(map(lambda x: float(x), 
    list(filter(lambda x: x.strip() != "", list(map(extract, result))))))

    return [dict(zip(order, result[idx:idx+4])) for idx in range(0,len(result),4)]

def convert_perf_time_video_len(args, perf_stats, reference_video_len=None):
    per_frame_execution_time = 0.0

    for stat in perf_stats:
        per_frame_execution_time += stat['preprocessing']
        per_frame_execution_time += stat['inference']
        per_frame_execution_time += stat['extract']
        per_frame_execution_time += stat['paint']

    return per_frame_execution_time / reference_video_len

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    logging.basicConfig(filename=args.output_log, filemode='w', level=logging.INFO)
    
    # Perform inference on the input stream
    counter = infer_on_stream(args, None)

    # perf_text = open(args.output_log, "r").read()

    # perf_stats = convert_perf_time(perf_text)

    # per_frame_execution_time = convert_perf_time_video_len(args, perf_stats, counter)

    # print(per_frame_execution_time)
    # print("FPS: ", 1 / per_frame_execution_time)

if __name__ == '__main__':
    main()
