#!/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""image enhance

这是数据增强的代码逻辑。
"""
import re
import PIL
import cv2
import yaml
import random
import logging

import numpy as np
from PIL import Image, ImageDraw
from typing import List

logger = logging.getLogger(__name__)

try:
    import blend_modes
except ModuleNotFoundError:
    logger.error(
        "blend_modes not found. Image Enhance is disabled. "
        "Please `pip install -r requirements.txt or update docker"
    )
    blend_modes = None


try:
    import imgaug as ia
    import imgaug.augmenters as iaa
except ModuleNotFoundError:
    logger.error(
        "imgaug not found. Image Enhance is disabled. "
        "Please `pip install -r requirements.txt or update docker"
    )
    ia = None
    iaa = None


class RandomSeedContext:
    """random context guard for imgaug"""

    def __init__(self, seed):
        self.seed = seed
        self.original_numpy_seed = None
        self.original_random_seed = None
        self.original_imgaug_seed = None

    def __enter__(self):
        # 保存原始的种子
        self.original_numpy_seed = np.random.get_state()
        self.original_random_seed = random.getstate()
        self.original_imgaug_seed = ia.random.get_global_rng().state

        # 设置新的种子
        np.random.seed(self.seed)
        random.seed(self.seed)
        ia.random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始的种子
        np.random.set_state(self.original_numpy_seed)
        random.setstate(self.original_random_seed)
        ia.random.get_global_rng().state = self.original_imgaug_seed


PATTERN = r"<ref>(.*?)</ref><quad>(.*?)</quad>"
PATTERN_QUAD = r"\((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\)"


def extract_quad(quad_str):
    """extract_quad"""
    matches = re.findall(PATTERN_QUAD, quad_str)
    macth = matches[0]
    return [(int(x), int(y)) for x, y in zip(macth[0::2], macth[1::2])]


def read_config(path):
    """read_config"""
    with open(path, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    return config


def draw_polygon_image(img: PIL.Image.Image, quads: List) -> PIL.Image.Image:
    """draw polygon image"""
    board_img = img.copy()
    draw = ImageDraw.Draw(board_img)
    for quad in quads:
        quad = [tuple(_) for _ in quad]
        color = tuple([255, 0, 0])
        draw.polygon(quad, outline="red")
    return board_img


def utils_to_bbox(quad):
    """utils to bbox"""
    topleft = np.amin(quad, axis=0)
    bottomright = np.amax(quad, axis=0)
    width, height = bottomright - topleft
    bbox = np.array([topleft[0], topleft[1], width, height], dtype=np.float32)
    return bbox


def utils_merge_quad(quads):
    """utils merge quad"""
    quads = np.array(quads, dtype=np.float32)
    topleft = np.amin(quads, axis=(0, 1))
    bottomright = np.amax(quads, axis=(0, 1))
    width, height = bottomright - topleft
    quad = np.array(
        [
            [topleft[0], topleft[1]],
            [topleft[0] + width, topleft[1]],
            [topleft[0] + width, topleft[1] + height],
            [topleft[0], topleft[1] + height],
        ],
        dtype=np.float32,
    )
    return quad


def utils_merge_bbox(bboxes):
    """utils merge bbox"""
    bboxes = np.array(bboxes, dtype=np.float32)
    topleft = np.amin(bboxes[..., :2], axis=0)
    bottomright = np.amax(bboxes[..., :2] + bboxes[..., 2:], axis=0)
    width, height = bottomright - topleft
    bbox = np.array([topleft[0], topleft[1], width, height], dtype=np.float32)
    return bbox


def utils_create_image(size, color=None):
    """
    Create an image with given size and color.

    :param size: The image size, as a 2-tuple (width, height)
    :type size: tuple
    :param color: The color of image, as a 4-tuple (RGBA)
    :type color: tuple, optional
    :return: RGBA image
    :rtype: Numpy array of float32 type
    """

    width, height = size
    out = np.zeros((int(height), int(width), 4), dtype=np.float32)
    if color is not None:
        out[...] = color
    return out


def utils_add_alpha_channel(image):
    """utils add alpha channel"""
    height, width, channel = image.shape
    if channel == 3:
        alpha = np.full((height, width, 1), 255, dtype=image.dtype)
        image = np.concatenate((image, alpha), axis=-1)
    return image


def utils_blend_image(src, dst, mode="normal", mask=False):
    """utils blend image"""
    alpha = dst[..., 3]

    if mode == "normal":
        src = Image.fromarray(src.astype(np.uint8))
        dst = Image.fromarray(dst.astype(np.uint8))
        out = Image.alpha_composite(dst, src)
        out = np.array(out, dtype=np.float32)
    else:
        blend = getattr(blend_modes, mode)
        out = blend(dst, src, 1)

    if mask:
        out[..., 3] = alpha
    return out


def utils_paste_image(src, dst, quad, mode="normal"):
    """utils paste image"""
    src_height, src_width = src.shape[:2]
    dst_height, dst_width = dst.shape[:2]
    origin = np.array(
        [[0, 0], [src_width, 0], [src_width, src_height], [0, src_height]],
        dtype=int,
    )
    quad = np.array(quad, dtype=int)

    src_topleft = np.amin(quad, axis=0)
    src_bottomright = np.amax(quad, axis=0)
    src_size = tuple(src_bottomright - src_topleft)
    dst_topleft = [0, 0]
    dst_bottomright = [dst_width, dst_height]
    dst_size = (dst_width, dst_height)

    topleft = np.amax([src_topleft, dst_topleft], axis=0)
    bottomright = np.amin([src_bottomright, dst_bottomright], axis=0)
    if any(topleft >= bottomright):
        return None

    if not all(
        (
            quad[0][0] == quad[3][0],
            quad[1][0] == quad[2][0],
            quad[0][1] == quad[1][1],
            quad[2][1] == quad[3][1],
            quad[1][0] - quad[0][0] == quad[2][0] - quad[3][0] == src_width,
            quad[3][1] - quad[0][1] == quad[2][1] - quad[1][1] == src_height,
        )
    ):
        origin = origin.astype(np.float32)
        quad = (quad - src_topleft).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(origin, quad)
        src = cv2.warpPerspective(src, matrix, src_size)

    sx, sy = np.clip(topleft - src_topleft, (0, 0), src_size)
    dx, dy = np.clip(bottomright - src_topleft, (0, 0), src_size)
    src_area = (slice(sy, dy), slice(sx, dx))

    sx, sy = np.clip(topleft - dst_topleft, (0, 0), dst_size)
    dx, dy = np.clip(bottomright - dst_topleft, (0, 0), dst_size)
    dst_area = (slice(sy, dy), slice(sx, dx))

    dst[dst_area] = utils_blend_image(src[src_area], dst[dst_area], mode=mode)


class MockGroup:
    """
    borrowed some code from https://github.com/clovaai/synthtiger/blob/master/synthtiger/layers/layer.py#L244
    """

    def __init__(self, quads):
        self.quads = quads
        self.bboxes = [utils_to_bbox(quad) for quad in quads]

    @property
    def quad(self):
        """quad"""
        return utils_merge_quad(self.quads)

    @property
    def bbox(self):
        """bbox"""
        return utils_merge_bbox(self.bboxes)

    @property
    def size(self):
        """size"""
        return np.array(self.bbox[2:])

    @property
    def center(self):
        """center"""
        return np.mean(self.quad, axis=0)


class Selector(object):
    """This is a samapler"""

    @classmethod
    def sample(cls, candidates: List, weights=None):
        """this is a function to  sample"""
        if weights is None:
            weights = [1] * len(candidates)
        prob = np.array(weights) / np.sum(weights)

        idx = np.random.choice(range(len(candidates)), replace=False, p=prob)
        sampled_candidate = candidates[idx]

        return sampled_candidate


class DocumentEffect:
    """
    文档变换效果, 旋转、仿射变换等
    Input:
        quads: List, 包含多组四点坐标
        meta = None: dict, 各变换效果配置参数, 默认None则随机生成
    Return:
        quads: List, 包含多组四点坐标
        meta: dict, 实际使用的配置参数
    """

    @classmethod
    def rotate_meta(cls, angle=(-45, 45), ccw=0):
        """generate rotate args"""
        meta = {
            "angle": np.random.uniform(angle[0], angle[1]),
            "ccw": np.random.rand() < ccw,
        }
        return meta

    @classmethod
    def rotate(cls, quads: List, meta: dict = None):
        """
        rotate the 4 points of the image
        """
        meta = cls.rotate_meta() if meta is None else meta

        angle = meta["angle"] * (1 if meta["ccw"] else -1)
        group = MockGroup(quads)
        matrix = cv2.getRotationMatrix2D(tuple(group.center), angle, 1)

        new_quads = []
        for quad in quads:
            new_quad = np.append(quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
            new_quad = new_quad.tolist()
            new_quads.append(new_quad)

        return new_quads, meta

    @classmethod
    def perspective_meta(
        cls, pxs=None, percents=None, aligns=((-1, 1), (-1, 1), (-1, 1), (-1, 1))
    ):
        """generate perspective args"""
        meta = {
            "pxs": pxs,
            "percents": tuple(
                np.random.uniform(percent[0], percent[1]) for percent in percents
            ),
            "aligns": tuple(np.random.uniform(align[0], align[1]) for align in aligns),
        }
        return meta

    @classmethod
    def perspective(cls, quads: List, meta: dict = None):
        """
        perspective the 4 points of the image
        """
        meta = cls.perspective_meta() if meta is None else meta

        pxs = meta["pxs"]
        percents = meta["percents"]
        aligns = meta["aligns"]

        aligns = np.tile(aligns, 4)[:4]
        if pxs is not None:
            pxs = np.tile(pxs, 4)[:4]
        if percents is not None:
            percents = np.tile(percents, 4)[:4]

        group = MockGroup(quads)
        sizes = np.tile(group.size, 2)
        new_sizes = np.tile(group.size, 2)

        if pxs is not None:
            new_sizes += pxs
        elif percents is not None:
            new_sizes *= percents

        values = (sizes - new_sizes) / 2
        aligns *= np.abs(values)
        offsets = [
            [values[0] + aligns[0], values[3] + aligns[3]],
            [-values[0] + aligns[0], values[1] + aligns[1]],
            [-values[2] + aligns[2], -values[1] + aligns[1]],
            [values[2] + aligns[2], -values[3] + aligns[3]],
        ]

        origin = group.quad
        quad = np.array(origin + offsets, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(origin, quad)

        new_quads = []
        for quad in quads:
            new_quad = np.append(quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
            new_quads.append(new_quad[..., :2] / new_quad[..., 2, np.newaxis])

        return new_quads, meta


class ImageEffect(object):
    """
    图像变换, 模糊, 色彩变换
    Input:
        images: np图像list
        meta = None: 变换参数
    Return:
        new_images: 变换后的图像list
        meta: 变换参数
    """

    @classmethod
    def elastic_distortion_meta(cls, alpha=(10, 15), sigma=(3, 3)):
        """generate elastic_distortion args"""
        meta = {
            "alpha": np.random.uniform(alpha[0], alpha[1]),
            "sigma": np.random.uniform(sigma[0], sigma[1]),
        }
        return meta

    @classmethod
    def elastic_distortion(cls, images: List[np.ndarray], meta: dict = None):
        """elastic_distortion"""
        if meta is None:
            meta = cls.elastic_distortion_meta()

        alpha = meta["alpha"]
        sigma = meta["sigma"]
        aug = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest")

        new_images = []
        for image in images:
            image = cls._preprocess_image(image)
            image = aug(image=image).astype(np.float32)
            new_images.append(image)
        return new_images, meta

    @classmethod
    def _preprocess_image(cls, npy_img):
        """preprocess image"""
        # npy_img = np.array(npy_img, dtype=np.float32)
        npy_img = utils_add_alpha_channel(npy_img)
        npy_img = npy_img.astype(np.uint8)
        return npy_img

    @classmethod
    def gaussian_blur_meta(cls, sigma=(1, 2)):
        """gaussian_blur_meta"""
        meta = {
            "sigma": np.random.uniform(sigma[0], sigma[1] + 1),
        }
        return meta

    @classmethod
    def gaussian_blur(cls, images: List[np.ndarray], meta: dict = None):
        """ """
        meta = cls.gaussian_blur_meta() if meta is None else meta

        sigma = meta["sigma"]
        aug = iaa.GaussianBlur(sigma=sigma)

        new_images = []
        for image in images:
            image = cls._preprocess_image(image)
            image = aug(image=image).astype(np.float32)
            new_images.append(image)
        return new_images, meta

    @classmethod
    def motion_blur_meta(cls, k=(3, 7), angle=(0, 360)):
        """motion_blur_meta"""
        meta = {
            "k": np.random.randint(k[0], k[1] + 1),
            "angle": np.random.uniform(angle[0], angle[1]),
        }
        return meta

    @classmethod
    def motion_blur(cls, images: List[np.ndarray], meta: dict = None):
        """ """
        meta = cls.motion_blur_meta() if meta is None else meta

        k = meta["k"]
        angle = meta["angle"]
        aug = iaa.MotionBlur(k=k, angle=angle)

        new_images = []
        for image in images:
            image = cls._preprocess_image(image)
            image = aug(image=image).astype(np.float32)
            new_images.append(image)
        return new_images, meta

    @classmethod
    def channel_shuffle_meta(cls):
        """channel_shuffle"""
        meta = {
            "channels_shuffled": np.random.permutation(np.arange(4)).tolist(),
        }
        return meta

    @classmethod
    def channel_shuffle(cls, images: List[np.ndarray], meta: dict = None):
        """channel_shuffle"""
        meta = cls.channel_shuffle_meta() if meta is None else meta
        channels_shuffled = meta["channels_shuffled"]

        new_images = []
        for image in images:
            if image.shape[-1] == 3:
                channels_shuffled = np.random.permutation(np.arange(3)).tolist()
            image = image[..., channels_shuffled]
            new_images.append(image)
        return new_images, meta

    @classmethod
    def brightness_meta(cls, beta=(-32, 32)):
        """brightness"""
        meta = {"beta": np.random.randint(beta[0], beta[1] + 1)}
        return meta

    @classmethod
    def brightness(cls, images: List[np.ndarray], meta: dict = None):
        """brightness"""
        meta = cls.brightness_meta() if meta is None else meta
        beta = meta["beta"]

        new_images = []
        for image in images:
            # print(image.dtype)
            image[..., :3] += beta
            image[..., :3] = np.clip(image[..., :3], 0, 255)
            new_images.append(image)
        return new_images, meta

    @classmethod
    def contrast_meta(cls, alpha=(0.5, 1.5)):
        """constrast_meta"""
        meta = {"alpha": np.random.uniform(alpha[0], alpha[1])}
        return meta

    @classmethod
    def contrast(cls, images: List[np.ndarray], meta: dict = None):
        """constrast"""
        meta = cls.contrast_meta if meta is None else meta
        alpha = meta["alpha"]

        new_images = []
        for image in images:
            image[..., :3] = alpha * image[..., :3] - 128 * (alpha - 1)
            image[..., :3] = np.clip(image[..., :3], 0, 255)
            new_images.append(image)
        return new_images, meta

    @classmethod
    def grayscale_meta(cls):
        """grayscale_meta"""
        return {}

    @classmethod
    def grayscale(cls, images: List[np.ndarray], meta: dict = None):
        """grayscale"""
        meta = cls.grayscale_meta() if meta is None else meta

        new_images = []
        for image in images:
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            image[..., :3] = gray[..., np.newaxis]
            new_images.append(image)
        return new_images, meta

    @classmethod
    def multiply_hue_meta(cls, hue=(0.5, 1.5)):
        """multiply_hue_meta"""
        meta = {"hue": np.random.uniform(hue[0], hue[1])}
        return meta

    @classmethod
    def multiply_hue(cls, images: List[np.ndarray], meta: dict = None):
        """multiply_hue"""
        meta = cls.multiply_hue_meta() if meta is None else meta
        hue = meta["hue"]
        aug = iaa.MultiplyHue(hue)

        new_images = []
        for image in images:
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = image.astype(np.uint8)
            image = aug(image=image).astype(np.float32)
            new_images.append(image)
        return new_images, meta

    @classmethod
    def multiply_saturation_meta(cls, saturation=(0.5, 1.5)):
        """multiply_saturation_meta"""
        meta = {"saturation": np.random.uniform(saturation[0], saturation[1])}
        return meta

    @classmethod
    def multiply_saturation(cls, images: List[np.ndarray], meta: dict = None):
        """multiply_saturation"""
        meta = cls.multiply_saturation_meta() if meta is None else meta
        saturation = meta["saturation"]
        aug = iaa.MultiplySaturation(saturation)

        new_images = []
        for image in images:
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = image.astype(np.uint8)
            image = aug(image=image).astype(np.float32)
            new_images.append(image)
        return new_images, meta

    @classmethod
    def cutout_meta(
        cls,
        nb_iterations=(1, 5),
        size=(0.01, 0.05),
        fill_mode="gaussian",
        fill_per_channel=True,
    ):
        """cutout_meta"""
        meta = {
            "nb_iterations": np.random.randint(nb_iterations[0], nb_iterations[1] + 1),
            "size": np.random.uniform(size[0], size[1]),
            "fill_mode": fill_mode,
            "fill_per_channel": fill_per_channel,
        }
        return meta

    @classmethod
    def cutout(cls, images: List[np.ndarray], meta: dict = None):
        """cutout"""
        meta = cls.cutout_meta() if meta is None else meta
        aug = iaa.Cutout(
            fill_mode=meta["fill_mode"],
            fill_per_channel=meta["fill_per_channel"],
            nb_iterations=meta["nb_iterations"],
            size=meta["size"],
        )

        new_images = []
        for image in images:
            image = cls._preprocess_image(image)
            image = aug(image=image).astype(np.float32)
            new_images.append(image)
        return new_images, meta


def paste2new_image(src_img: PIL.Image.Image, new_w, new_h, new_img_quad) -> np.ndarray:
    """
    将变换后的图片放置到新的图像上, 并添加白色背景
    """
    src_img = src_img.convert("RGBA")
    np_img = np.array(src_img)  # np, RGBA
    new_img = utils_create_image((new_w, new_h))  # np, RGBA
    utils_paste_image(np_img, new_img, new_img_quad)

    # Add white background
    white_bg = np.ones_like(new_img) * 255
    new_img = np.where(new_img[:, :, 3:] > 0, new_img, white_bg)
    return new_img


def apply_effect(img: PIL.Image.Image, effect_augs: List[List]):
    """apply effect"""
    for effect in effect_augs:
        if effect["func_name"] == "paste2new_image":
            img = paste2new_image(img, **effect["args"])
        else:
            cls_name, func_name = effect["func_name"].split(".")
            seed = effect["seed"]
            if isinstance(img, PIL.Image.Image):
                img = np.array(img).astype(np.float32)

            with RandomSeedContext(seed):
                img = getattr(eval(cls_name), func_name)([img], effect["args"])[0][0]

    if isinstance(img, np.ndarray):
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img = img.convert("RGB")
    return img
