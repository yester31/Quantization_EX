#  by yhpark 2023-07-17
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import os
import torchvision.transforms as transforms
from PIL import Image


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, TRT_LOGGER, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.TRT_LOGGER = TRT_LOGGER
        self.batch_allocation = None
        self.batch_size = None
        self.img_dir = None
        self.max_num_images = None
        self.file_list = None
        self.img_count = 0

    def set_calibrator(self, batch_size, shape, dtype, img_dir, max_num_images=None):
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.max_num_images = max_num_images
        size = int(np.dtype(dtype).itemsize * np.prod(shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.file_list = os.listdir(img_dir)
        self.max_img_size = len(os.listdir(img_dir))
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.batch_size:
            return self.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        try:
            if self.max_img_size - 1 == self.img_count:
                print("Finished calibration batches")
                return None

            calib_data_name = self.file_list[self.img_count]
            calib_data_path = self.img_dir + "/" + calib_data_name
            print(f"[{self.img_count}] calib data load... {calib_data_path} ")
            img = Image.open(calib_data_path)
            self.img_count += 1
            if img.mode == "RGB":
                tensor = self.transform(img)
                batch = np.array(tensor, dtype=np.float32, order="C")
                cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
                return [int(self.batch_allocation)]
            else:
                calib_data_name = self.file_list[self.img_count]
                calib_data_path = self.img_dir + "/" + calib_data_name
                print(f"[{self.img_count}] calib data load... {calib_data_path} ")
                img = Image.open(calib_data_path)
                self.img_count += 1
                tensor = self.transform(img)
                batch = np.array(tensor, dtype=np.float32, order="C")
                cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
                return [int(self.batch_allocation)]

        except StopIteration:
            print("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            print("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)
