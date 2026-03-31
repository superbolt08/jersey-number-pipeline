# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import imgaug.augmenters as iaa
import numpy as np
from PIL import ImageFilter, Image, ImageEnhance
import random
from timm.data import auto_augment

from strhub.data import aa_overrides

aa_overrides.apply()

_OP_CACHE = {}


def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))


def gaussian_blur(img, radius, **__):
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)


def motion_blur(img, k, **__):
    k = _get_param(k, img, 0.08, 3) | 1  # bin to odd values
    key = 'motion_blur_' + str(k)
    op = _get_op(key, lambda: iaa.MotionBlur(k))
    return Image.fromarray(op(image=np.asarray(img)))


def gaussian_noise(img, scale, **_):
    scale = _get_param(scale, img, 0.25) | 1  # bin to odd values
    key = 'gaussian_noise_' + str(scale)
    op = _get_op(key, lambda: iaa.AdditiveGaussianNoise(scale=scale))
    return Image.fromarray(op(image=np.asarray(img)))


def poisson_noise(img, lam, **_):
    lam = _get_param(lam, img, 0.2) | 1  # bin to odd values
    key = 'poisson_noise_' + str(lam)
    op = _get_op(key, lambda: iaa.AdditivePoissonNoise(lam))
    return Image.fromarray(op(image=np.asarray(img)))


def occlusion(img, scale, **_):
    scale = _get_param(scale, img, 0.25)
    img_np = np.array(img).copy()

    h, w = img_np.shape[:2]
    occ_w = int(w * np.random.uniform(0.1, 0.3))
    occ_h = int(h * np.random.uniform(0.2, 0.6))

    x = np.random.randint(0, w - occ_w)
    y = np.random.randint(0, h - occ_h)

    img_np[y:y+occ_h, x:x+occ_w] = np.random.randint(0, 255)
    return Image.fromarray(img_np)

def color_jitter(img, level, **_):
    factor = 1.0 + (level / 10.0) #Sets the scale of jitter between 1 and 2
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0, factor))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0, factor))
    if random.random() < 0.8:
        img = ImageEnhance.Color(img).enhance(random.uniform(1.0, factor))

    return img


def perspective_warp(img, scale, **_):
    scale = _get_param(scale, img, 0.15)

    key = 'perspective_' + str(scale)
    op = _get_op(key, lambda: iaa.PerspectiveTransform(scale=(0.01, scale / 100.0)))

    return Image.fromarray(op(image=np.asarray(img)))

def elastic_distortion(img, alpha, **_):
    alpha = _get_param(alpha, img, 0.10)

    key = 'elastic_' + str(alpha)
    op = _get_op(key, lambda: iaa.ElasticTransformation(
        alpha=alpha,
        sigma=alpha * 0.5
    ))

    return Image.fromarray(op(image=np.asarray(img)))

def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return level,


_RAND_TRANSFORMS = auto_augment._RAND_INCREASING_TRANSFORMS.copy()
_RAND_TRANSFORMS.remove('SharpnessIncreasing')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    # 'MotionBlur',
    # 'GaussianNoise',
    'PoissonNoise',
    'Oclussion',
    'ColorJitter',
    'ElasticDistortion',
    'PerspectiveWarp'
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur': partial(_level_to_arg, max=4),
    'MotionBlur': partial(_level_to_arg, max=20),
    'GaussianNoise': partial(_level_to_arg, max=0.1 * 255),
    'PoissonNoise': partial(_level_to_arg, max=40),
    'Oclussion':partial(_level_to_arg, max=50),
    'ColorJitter': partial(_level_to_arg, max=10),
    'PerspectiveWarp': partial(_level_to_arg, max=20),
    'ElasticDistortion': partial(_level_to_arg, max=8)
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise,
    'Oclussion':occlusion,
    'ColorJitter':color_jitter,
    'PerspectiveWarp': perspective_warp,
    'ElasticDistortion': elastic_distortion,
})


def rand_augment_transform(magnitude=9, num_layers=5):
    # These are tuned for magnitude=5, which means that effective magnitudes are half of these values.
    hparams = {
        'rotate_deg': 30,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.2,
        'translate_x_pct': 0.10,
        'translate_y_pct': 0.30
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude, hparams=hparams, transforms=_RAND_TRANSFORMS)
    # Supply weights to disable replacement in random selection (i.e. avoid applying the same op twice)
    choice_weights = [1. / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)

