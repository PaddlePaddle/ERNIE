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
""" Download utils """

import base64
import datetime
import hashlib
import io
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from PIL.ExifTags import TAGS

RAW_VIDEO_DIR = "./download_tmp/raw_video/"
RAW_IMAGE_DIR = "./download_tmp/raw_images/"
EXTRACTED_FRAME_DIR = "./download_tmp/extracted_frames/"
TMP_DIR = "./download_tmp/upload_tmp/"


def file_download(url, download_dir, save_to_disk=False, retry=0, retry_interval=3):
    """
    Description: Download url, if url is PIL, return directly
    Args:
        url(str, PIL): http/local path/io.Bytes, note that io.Bytes is the image byte stream
        download_path: when save_to_disk=True, return the saved address
        save_to_disk: whether to save in the local path

    """
    from data_processor.utils.video_utils import VideoReaderWrapper

    if isinstance(url, Image.Image):
        return url
    elif isinstance(url, VideoReaderWrapper):
        return url
    elif url.startswith("http"):
        response = requests.get(url)
        bytes_data = response.content
    elif os.path.isfile(url):
        if save_to_disk:
            return url
        bytes_data = open(url, "rb").read()
    else:
        bytes_data = base64.b64decode(url)
    if not save_to_disk:
        return bytes_data

    download_path = os.path.join(download_dir, get_filename(url))
    Path(download_path).parent.mkdir(parents=True, exist_ok=True)
    with open(download_path, "wb") as f:
        f.write(bytes_data)
    return download_path


def get_filename(url=None):
    """
    Get Filename
    """
    if url is None:
        return str(uuid.uuid4()).replace("-", "")
    t = datetime.datetime.now()
    if not isinstance(url, bytes):
        url = url.encode("utf-8")

    md5_hash = hashlib.md5(url).hexdigest()
    pid = os.getpid()
    tid = threading.get_ident()

    # Remove the suffix to prevent save-jpg from reporting errors
    image_filname = f"{t.year}-{t.month:02d}-{t.day:02d}-{pid}-{tid}-{md5_hash}"
    return image_filname


def get_downloadable(url, download_dir=RAW_VIDEO_DIR, save_to_disk=False, retry=0, retry_interval=3):
    """download video and store it in the disk

    return downloaded **path** if save_to_disk is set to true
    return downloaded **bytes** if save_to_disk is set to false
    """

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    downloaded_path = file_download(
        url,
        download_dir,
        save_to_disk=save_to_disk,
        retry=retry,
        retry_interval=retry_interval,
    )
    return downloaded_path


def get_downloadable_image(download_path, need_exif_info, retry_max_time=0, retry_interval=3):
    """
    Get downloadable with exif info and image processing
    """

    def get_image_exif(image):
        exif_data = image._getexif()
        exif_info = {}
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif_info[tag_name] = value.strip()
        return exif_info

    def has_transparent_background(img):
        """has_transparent_background"""
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Check for any pixel with alpha channel less than 255 (fully opaque)
            alpha = img.convert("RGBA").split()[-1]
            if alpha.getextrema()[0] < 255:
                return True
        return False

    def add_white_background(img):
        """
        Add a white background to a transparent background image
        """
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # Create an image with a white background and the same size as the original image
        img_white_background = Image.new("RGBA", img.size, (255, 255, 255))

        # Paste the original image onto a white background
        img_white_background.paste(img, (0, 0), img)

        return img_white_background

    def change_I16_to_L(img):
        """
        Convert image from I;16 mode to L mode
        """
        # Since the point function in I mode only supports addition, subtraction, and multiplication, the following * (1 / 256) cannot be changed to division.
        return img.point(lambda i: i * (1 / 256)).convert("L")

    image = get_downloadable(download_path, save_to_disk=False, retry=retry_max_time, retry_interval=retry_interval)
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.open(io.BytesIO(image))
    if need_exif_info:
        try:
            exif_info = get_image_exif(pil_image)
        except Exception as why:
            exif_info = {}
    else:
        exif_info = {}

    try:
        if pil_image.mode == "I;16":
            pil_image = change_I16_to_L(pil_image)
        if has_transparent_background(pil_image):
            pil_image = add_white_background(pil_image)
    except Exception as e:
        pass

    return pil_image.convert("RGB"), exif_info


def str2hash(url):
    """
    str2hash
    """
    return hashlib.sha256(url.encode()).hexdigest()


def pil2hash(pil):
    """
    PIL.Image to hash
    """
    byte_io = io.BytesIO()
    pil.save(byte_io, format="PNG")  # avoid compression effects
    image_bytes = byte_io.getvalue()

    return hashlib.sha256(image_bytes).hexdigest()


def imagepath_to_base64(image_path):
    """imagepath_to_base64"""
    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    return base64_encoded


def pil_image_to_base64(image):
    """pil_image_to_base64"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    return base64_encoded


def http_to_pil_image(url):
    """http_to_pil_image"""
    response = requests.get(url)
    image_data = io.BytesIO(response.content)
    pil_image = Image.open(image_data).convert("RGB")
    return pil_image


def http_to_image_base64(url):
    """http_to_image_base64"""
    response = requests.get(url)
    image_data = io.BytesIO(response.content)
    return base64.b64encode(image_data.getvalue()).decode("utf-8")


def base64_to_pil_image(base64_string):
    """ " base64_to_pil_image"""
    image_bytes = base64.b64decode(base64_string)
    buffer = io.BytesIO(image_bytes)
    image = Image.open(buffer)
    return image


def get_hashable(to_be_hashed):
    """get hashable"""
    if isinstance(to_be_hashed, bytes):
        return to_be_hashed
    elif isinstance(to_be_hashed, Image.Image):
        return to_be_hashed.tobytes()
    elif isinstance(to_be_hashed, str):
        return to_be_hashed.encode("utf-8")
    else:
        raise ValueError(f"not support type: {type(to_be_hashed)}")


def load_dict_from_npz(npzfile):
    """load_dict_from_npz"""
    with np.load(npzfile, allow_pickle=True) as data:
        loaded_dict = {key: data[key] for key in data.files}
    return loaded_dict


def image_info_2_hash(img_one):
    """
    image info to hash
    """
    if isinstance(img_one["image_url"], str):
        return str2hash(img_one["image_url"])
    elif isinstance(img_one["image_url"], Image.Image):
        return pil2hash(img_one["image_url"])
    else:
        raise ValueError("only support str or PIL.Image now.")
