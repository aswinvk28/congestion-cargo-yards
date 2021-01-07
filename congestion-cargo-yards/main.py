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
from yolov3_module import *

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
    parser.add_argument("--dict_export", help="", type=bool, default=False)
    parser.add_argument("--keep_aspect_ratio", help="", type=bool, default=True)
    parser.add_argument("--yolo", help="", type=bool, default=False)
    parser.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
            "detections filtering", default=0.4, type=float)
    parser.add_argument("--relative_overlap_area", help="Optional. IoU relative overlap area", default=0.4, type=float)
    parser.add_argument("--relative_union_area", help="Optional. IoU relative union area", default=0.4, type=float)
    parser.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
            default=0.5, type=float)
    
    return parser


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

def finalize_draw_boxes(boxes, frame, args):
    iou = []
    for ii, box_1 in enumerate(boxes):
        for box_2 in boxes[ii+1:]:
            _ratio, _num, _denom = intersection_over_union(
                dict(zip(['xmin', 'ymin', 'xmax', 'ymax'], box_1)), 
                dict(zip(['xmin', 'ymin', 'xmax', 'ymax'], box_2)))
            iou.append([_ratio, _num, _denom])
    iou = np.array(iou)
    iou[:,1] /= iou[:,1].max()
    iou[:,2] /= iou[:,2].max()
    idx = np.where((iou[:,0] >= args.iou_threshold) & (iou[:,1] <= args.relative_overlap_area) \
        & (iou[:,2] <= args.relative_union_area))[0]
    idx = np.floor(1 + np.sqrt(1+idx*2*4)) / 2
    idx = np.unique(idx).astype(np.int32)
    print(idx)
    boxes = np.array(boxes)
    for box in boxes[idx]:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, args.th)

    return frame

def draw_boxes(frame, result, args, width, height):
    confs = []
    boxes = []
    for box in result[0][0]:
        conf = box[2]
        if conf >= args.pt:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            confs.append(conf)
            boxes.append((xmin, ymin, xmax, ymax))
            
    return frame, confs, boxes

def infer_on_batch_result(infer_network, frames, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    if args.dict_export:
        results = infer_network.extract_outputs_dict(request_id=request_id)
    else:
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
        elif args.yolo:
            frame, confs = draw_yolo_bounding_boxes(results, infer_network.network, 
            infer_network.get_input_shape()[1], infer_network.get_input_shape()[0], frame, args)
        else:
            frame, confs, boxes = draw_boxes(frame, results[ii,:,:,:], args, width, height)
            frame = finalize_draw_boxes(boxes, frame, args)
        frames[ii] = frame
        confidences = np.append(confidences, confs).tolist()
    
    # Write out the frame
    
    return frames, thrs, confidences

def infer_on_multi_result(infer_network, frames, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    if args.dict_export:
        results = infer_network.extract_outputs_dict(request_id=request_id)
    else:
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
        elif args.yolo:
            frame, confs = draw_yolo_bounding_boxes(results, infer_network.network, 
            infer_network.get_input_shape()[1], infer_network.get_input_shape()[0], frame, args)
        else:
            frame, confs, boxes = draw_boxes(frame, results[:,:,output_shape*ii:output_shape*ii+output_shape,:], 
        args, width, height)
            frame = finalize_draw_boxes(boxes, frame, args)
        frames[ii] = frame
        confidences = np.append(confidences, confs).tolist()
    
    # Write out the frame
    
    return frames, thrs, confidences

def infer_on_result(infer_network, frame, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    if args.dict_export:
        results = infer_network.extract_outputs_dict(request_id=request_id)
    else:
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
    elif args.yolo:
        frame, confs = draw_yolo_bounding_boxes(results, infer_network.network, 
        infer_network.get_input_shape()[3], infer_network.get_input_shape()[2], frame, args)
    else:
        frame, confs, boxes = draw_boxes(frame, results, args, width, height)
        frame = finalize_draw_boxes(boxes, frame, args)
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
