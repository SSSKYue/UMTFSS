import random
import numpy as np
import cv2

import torch

manual_seed = 123
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)


class Compose(object):

    def __init__(self, segtransform, mean, std):
        self.segtransform = segtransform
        self.totensor = ToTensor()
        self.normalise = Normalize(mean, std)

    def __call__(self, image, additional_label=None):
        if len(self.segtransform) > 0:
            for t in self.segtransform:
                image, additional_label = t(image, additional_label)
        image, additional_label = self.totensor(image, additional_label)
        image = self.normalise(image)
        return image, additional_label


class ToTensor(object):
    """
    numpy array -> torch.Tensor
    """
    @staticmethod
    def transform_image(image):
        if not isinstance(image, np.ndarray):
            raise RuntimeError(f"ToTensor() only handle np.ndarray, got image of type {type(image)}.\n")
        if len(image.shape) != 3:
            raise RuntimeError(f"ToTensor() only handle image of shape (H, W, C), got image of shape {image.shape}.\n")

        image = torch.from_numpy(
            image.transpose(
                (2, 0, 1)
            )
        )
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image

    @staticmethod
    def transform_label(label):
        if not isinstance(label, np.ndarray):
            raise RuntimeError(f"ToTensor() only handle np.ndarray, got label of type {type(label)}.\n")
        if len(label.shape) != 2:
            raise (RuntimeError(f"ToTensor() only handle label of shape (H, W), got label of shape {label.shape}.\n"))

        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return label

    def __call__(self, image, additional_label=None):
        image = self.transform_image(image)
        # label = self.transform_label(label)

        if additional_label is not None:
            new_additional_label = []
            for l in additional_label:
                new_additional_label.append(self.transform_label(l))
            additional_label = new_additional_label
        return image, additional_label


class Normalize(object):
    """
    Normalise image
    """
    def __init__(self, mean, std):
        if not len(mean) == 3 and len(std) == 3:
            raise RuntimeError(f'Normalise() expected mean and std of length 3, got mean={mean}, std={std}. \n')
        self.mean = mean
        self.std = std

    def __call__(self, image):
        """
        Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
        :param image: torch.FloatTensor of shape (C, H, W)
        :return:
        """
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image


class Resize(object):
    """
    Resize input to given size
    """
    def __init__(self, size):
        """
        :param size: int
        """
        self.size = size

    @staticmethod
    def find_new_hw(ori_h, ori_w, test_size):
        if ori_h >= ori_w:
            ratio = test_size * 1.0 / ori_h
            new_h = test_size
            new_w = int(ori_w * ratio)
        else:
            ratio = test_size * 1.0 / ori_w
            new_h = int(ori_h * ratio)
            new_w = test_size

        if new_h % 8 != 0:
            new_h = int(new_h / 8) * 8
        else:
            new_h = new_h
        if new_w % 8 != 0:
            new_w = int(new_w / 8) * 8
        else:
            new_w = new_w
        return int(new_h), int(new_w)

    def resize_image(self, image, new_h, new_w):
        image_crop = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
        new_image = np.zeros((self.size, self.size, 3))
        new_image[:new_h, :new_w, :] = image_crop
        return new_image

    def resize_mask(self, mask, new_h, new_w):
        mask = cv2.resize(mask.astype(np.float32), dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
        new_mask = np.ones((self.size, self.size)) * 255
        new_mask[:new_h, :new_w] = mask
        return new_mask

    def __call__(self, image, additional_label=None):
        new_h, new_w = self.find_new_hw(image.shape[0], image.shape[1], self.size)

        image = self.resize_image(image, new_h=new_h, new_w=new_w)
        return image, additional_label


class RandScale(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    """
    def __init__(self, scale):
        if len(scale) !=2:
            raise RuntimeError(f"RandScale() expecting scale=(min, max), got scale={scale}")
        if len(scale) ==2 and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise RuntimeError(f"RandScale() expecting scale=(min, max), got scale={scale}")

    def __call__(self, image, additional_label=None):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio

        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        if additional_label is not None:
            new_additional_label = []
            for l in additional_label:
                l = cv2.resize(l, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
                new_additional_label.append(l)
            additional_label = new_additional_label
        return image, additional_label


class Crop(object):
    """
    Crops the given numpy array to given size.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.size=size
        else:
            raise RuntimeError(f"Crop() expect size to be int, got {size}")

        if crop_type in ['center', 'rand']:
            self.crop_type = crop_type
        else:
            raise (RuntimeError(f"Crop() expect crop type to be rand or center, got {crop_type}\n"))

        if len(padding) == 3:
            self.padding = padding
        else:
            raise (RuntimeError(f"Crop() expect len(padding)=3, got {padding}\n"))

        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise RuntimeError(f"Crop() expect size to be int, got {ignore_label}")

    def pad(self, image, label, additional_label=None):
        h, w = label.shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(
                image,
                pad_h_half, pad_h - pad_h_half,
                pad_w_half, pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.padding
            )
            label = cv2.copyMakeBorder(
                label,
                pad_h_half, pad_h - pad_h_half,
                pad_w_half, pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.ignore_label
            )
            if additional_label is not None:
                new_additional_label = []
                for l in additional_label:
                    l = cv2.copyMakeBorder(
                        l,
                        pad_h_half, pad_h - pad_h_half,
                        pad_w_half, pad_w - pad_w_half,
                        cv2.BORDER_CONSTANT,
                        value=self.ignore_label
                    )
                    new_additional_label.append(l)
                additional_label = new_additional_label
        return image, label, additional_label

    def crop(self, image, additional_label=None):
        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.size)
            w_off = random.randint(0, w - self.size)
        else:
            h_off = int((h - self.size) / 2)
            w_off = int((w - self.size) / 2)
        image = image[h_off:h_off+self.size, w_off:w_off+self.size]
        # label = label[h_off:h_off+self.size, w_off:w_off+self.size]
        if additional_label is not None:
            new_additional_label = []
            for l in additional_label:
                l = l[h_off:h_off+self.size, w_off:w_off+self.size]
                new_additional_label.append(l)
            additional_label = new_additional_label
        return image, additional_label

    def __call__(self, image, additional_label=None):
        raw_image, raw_label, raw_additional_label = self.pad(image, label, additional_label)
        pos_num = 0
        raw_pos_num = np.sum(raw_label == 1)
        crop_cnt = 0

        while (pos_num < 0.85 * raw_pos_num) and (crop_cnt <= 30):
            image, label, additional_label = self.crop(raw_image, raw_label, raw_additional_label)
            raw_pos_num = np.sum(raw_label == 1)
            pos_num = np.sum(label == 1)  
            crop_cnt += 1

        if image.shape != (self.size, self.size, 3):
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            if additional_label is not None:
                new_additional_label = []
                for l in additional_label:
                    l = cv2.resize(l, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    new_additional_label.append(l)
                additional_label = new_additional_label
        return image, additional_label


class RandRotate(object):
    """
    Randomly rotate image & label with rotate factor
    """
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        if len(rotate) == 2 and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError(f"RandRotate() scale param error, got rotate={rotate}\n"))

        if len(padding) == 3:
            self.padding = padding
        else:
            raise (RuntimeError(f"RandRotate() expect len(padding)=3, got {padding}\n"))

        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise RuntimeError(f"RandRotate() expect size to be int, got {ignore_label}")

        self.p = p

    def __call__(self, image, additional_label=None):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            if additional_label is not None:
                new_additional_label = []
                for l in additional_label:
                    l = cv2.warpAffine(l, matrix, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
                    new_additional_label.append(l)
                additional_label = new_additional_label
        return image, additional_label


class RandomHorizontalFlip(object):
    """
    Randomly flip the image horizontally
    """
    def __init__(self, p=0.5):
        """
        :param p: probability of flip
        """
        self.p = p

    def __call__(self, image, additional_label=None):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            if additional_label is not None:
                new_additional_label = []
                for l in additional_label:
                    l = cv2.flip(l, 1)
                    new_additional_label.append(l)
                additional_label = new_additional_label
        return image, additional_label


class RandomGaussianBlur(object):
    """
    Gaussian blur the image randomly
    """
    def __init__(self, radius=5):
        """
        :param radius: radius of Gaussian kernel
        """
        self.radius = radius

    def __call__(self, image, additional_label=None):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, additional_label
