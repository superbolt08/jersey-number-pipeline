# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import json
import sys
import time

ROOT = './pose/ViTPose/'
sys.path.append(str(ROOT))  # add ROOT to PATH

from argparse import ArgumentParser

import torch

# xtcocotools has no reliable Py3.12 wheels; pycocotools exposes the same COCO API for this script.
try:
    from xtcocotools.coco import COCO
except ImportError:
    from pycocotools.coco import COCO

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


class _StepTimer:
    """Delta since last log line and total elapsed (matches main.py SoccerNet pipeline)."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0

    def tick(self, msg):
        now = time.perf_counter()
        dt = now - self._last
        total = now - self._t0
        self._last = now
        print(f"{msg}  [+{dt:.2f}s | {total:.1f}s total]", flush=True)


def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--out-json',
        type=str,
        default='',
        help='Json file containing results.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
             'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()
    timer = _StepTimer()

    device = args.device.lower()
    if device.startswith('cuda') and not torch.cuda.is_available():
        warnings.warn(
            'CUDA was requested but this PyTorch build has no CUDA (or no GPU driver). '
            'Using CPU for ViTPose; inference will be slow.',
            UserWarning,
        )
        device = 'cpu'

    coco = COCO(args.json_file)
    img_keys = list(coco.imgs.keys())
    n_img = len(img_keys)
    timer.tick(f'ViTPose: {n_img} image(s), device={device}')
    timer.tick('ViTPose: loading model (slow on first run / CPU)')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=device)
    timer.tick('ViTPose: model ready. Starting inference.')

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    results = []

    indices = range(len(img_keys))
    if tqdm is not None:
        indices = tqdm(
            indices,
            desc='ViTPose',
            unit='img',
            mininterval=0.5,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        )
    # process each image
    for i in indices:
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # print(pose_results)
        results.append(
            {"img_name": image['file_name'], "id": image_id, "keypoints": pose_results[0]['keypoints'].tolist()})

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)

        if tqdm is None and (n_img <= 20 or (i + 1) % max(1, n_img // 10) == 0 or i == n_img - 1):
            timer.tick(f'ViTPose: finished {i + 1}/{n_img} image(s)')

    if args.out_json != '':
        with open(args.out_json, 'w') as fp:
            json.dump({"pose_results": results}, fp)
        timer.tick(f'ViTPose: wrote {args.out_json}')
    timer.tick('ViTPose: done.')


if __name__ == '__main__':
    main()