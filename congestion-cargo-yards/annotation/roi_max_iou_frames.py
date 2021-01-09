import numpy as np
import cv2
import copy
from yolov3_module import intersection_over_union

def check_roi(boxes, current, previous, permanent_ids, temp_ids, previous_temp_ids):
  if len(temp_ids)==0:
    current = c

    for box in boxes["frame_"+str(c)]['rois']:
      temp_ids[str(temp_queue[0])] = {'box': box, 'frame_count': 0, 'same_count': 0}
      temp_queue.remove(temp_queue[0])

  else:
    previous_temp_ids = copy.deepcopy(temp_ids)
    previous = current
    current = c
    for box in boxes["frame_"+str(current)]['rois']:
      box_1 = {'ymin': box[0],'xmin': box[1],'ymax': box[2],'xmax': box[3]}

      max_iou = 0
      max_iou_index = -1
      for ids in temp_ids:
        previous_box = temp_ids[ids]['box']
        box_2 = {'ymin': previous_box[0],'xmin': previous_box[1],'ymax': previous_box[2],'xmax': previous_box[3]}
        t_iou = intersection_over_union(box_1, box_2)
        if t_iou > max_iou:
          max_iou = t_iou
          max_iou_index = ids
      
      #print(max_iou, max_iou_index)
      
      if max_iou>args.max_iou:
        temp_ids[max_iou_index]['frame_count'] += 1
        temp_ids[max_iou_index]['box'] = box

      else:
        temp_ids[str(temp_queue[0])] = {'box': box, 'frame_count': 0, 'same_count': 0}
        temp_queue.remove(temp_queue[0])


    pop_ids = []
    for ids in temp_ids:
      if ids in previous_temp_ids:
        if temp_ids[ids]['frame_count']==previous_temp_ids[ids]['frame_count']:
          temp_ids[ids]['same_count'] += 1
        else:
          temp_ids[ids]['same_count'] = 0
      

      if temp_ids[ids]['frame_count']==frame_count_to_be_permanent and ids not in permanent_ids:
        permanent_ids[ids] = {}
        permanent_ids[ids]['vanish_count'] = 0


      if temp_ids[ids]['same_count']==count_to_vanish:
        #remove temp_ids
        pop_ids += [ids]

    for ids in pop_ids:
      if ids in permanent_ids:
        permanent_ids.pop(ids)
      temp_queue += [ids]
      xz = temp_ids.pop(ids)

    return current, previous, permanent_ids, temp_ids, previous_temp_ids