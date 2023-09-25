import cv2
import numpy as np
import random
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import make_dataset, find_classes, IMG_EXTENSIONS

from typing import Callable, Dict, List, Optional, Tuple

class DGDataset(VisionDataset):

    def __init__(self, root, apply_transform=False, reconstruction=False):

        super(DGDataset, self).__init__(root, transform=None)
        self.apply_transform = apply_transform
        self.reconstruction = reconstruction

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        sample = self.load_image(path)

        if self.apply_transform:
            sample = self.custom_transform(sample)
        else:
            sample = self.custom_transform_val(sample)

        if self.reconstruction:
            canny_sample = self.extract_canny(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.reconstruction:
            return sample, canny_sample, target, path
        else:
            return sample, target, path

    @staticmethod
    def load_image(path):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformer = transforms.ToTensor()
        img_rgb = transformer(img_rgb)
        return img_rgb

    @staticmethod
    def extract_canny(img):

        img = img.numpy()
        img = img * 255
        img = img.astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        # print("original")
        # print(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print("gray")
        # print(img_gray)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        lower = 110
        upper = 180
        edges = cv2.Canny(image=img_blur, threshold1=lower, threshold2=upper)
        # print("edges")
        # print(edges)
        transformer = transforms.ToTensor()
        edges = transformer(edges)
        return edges

    # @staticmethod
    # def custom_transform(image):
    #     # Resize2
    #     resize = transforms.Resize(size=(256, 256))
    #     image = resize(image)
    #     # Random crop
    #     i, j, h, w = transforms.RandomCrop.get_params(
    #         image, output_size=(224, 224))
    #     image = F.crop(image, i, j, h, w)

    #     # Random horizontal flipping
    #     if random.random() > 0.5:
    #         image = F.hflip(image)

    #     return image
    
    @staticmethod
    def custom_transform(image):
        
        # Define the transformation pipeline
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Apply transformations
        image = augment_transform(image)
        return image
    
    
    @staticmethod
    def custom_transform_val(image):

        # Resize2
        resize = transforms.Resize(size=(256, 256))
        ccrop = transforms.CenterCrop(size=(224, 224))

        image = resize(image)
        image = ccrop(image)

        return image

    def __len__(self) -> int:
        return len(self.samples)
