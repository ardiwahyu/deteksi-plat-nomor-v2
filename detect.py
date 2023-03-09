from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from pathlib import Path
import argparse
import os
import platform
import sys
import torch

def run_detection(model,source,int_color,size=640):
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((size,size), s=stride)
    bs = 1
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, 0.5, 0.45, None, False, max_det=1000)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        array_xyxy = []
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_dir = increment_path(Path('yolov5/runs/detect') / 'exp', exist_ok=True)
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if False else im0  # for save_crop
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                    array_xyxy.append(xyxy)
                    annotator.box_label(xyxy, label, color=colors(int_color, True))

            # Stream results
            im0 = annotator.result()
            return array_xyxy, im0