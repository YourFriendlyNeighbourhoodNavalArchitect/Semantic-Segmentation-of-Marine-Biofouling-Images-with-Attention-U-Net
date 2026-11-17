import numpy as np
import albumentations as A
from torch import from_numpy, float32
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToImage, ToDtype

class SegmentationDataset(Dataset):
    def __init__(self, imageFolder, maskFolder, IDs, resolution, classRemap, augmentationFlag, seed):
        self.IDs = list(IDs)
        self.augmentationFlag = augmentationFlag

        # Implement lazy loading of files to reduce computational overhead.
        # Images and masks are paired by ID, so that a missing file cannot misalign the remaining pairs.
        self.dataset = [(imageFolder / f'{ID}.jpg', maskFolder / f'{ID}.npy') for ID in self.IDs]
        missingFiles = [path.name for pair in self.dataset for path in pair if not path.exists()]
        if missingFiles:
            raise FileNotFoundError(f'Files not found in {imageFolder.parent}: {", ".join(missingFiles)}.')

        # Annotation classes are mapped to model classes upon loading, leaving the stored masks untouched.
        self.remap = np.array(classRemap, dtype = np.uint8)

        # Define transformation pipeline. Its own random generator is seeded, so that augmentations are reproducible
        # even without data loading workers. Workers reseed it in turn (see seedWorker).
        # Training crops are taken from the image resized to twice the input resolution, so that the network sees the image
        # with the same aspect ratio as during validation and testing, at up to twice the magnification.
        if self.augmentationFlag:
            self.transformCompose = A.Compose([A.Resize(2*resolution[0], 2*resolution[1]),
                                               A.RandomSizedCrop(min_max_height = (resolution[0], 2*resolution[0]), size = resolution, p = 1.0),
                                               # Hull images have no preferred orientation.
                                               A.RandomRotate90(p = 1.0),
                                               A.HorizontalFlip(p = 0.5),
                                               A.ColorJitter(brightness = 0.05, contrast = 0.05, saturation = 0.05, hue = 0.05, p = 0.5),
                                               A.MotionBlur(blur_limit = (3, 9), p = 0.5)], seed = seed)
        else:
            self.transformCompose = A.Compose([A.Resize(resolution[0], resolution[1])], seed = seed)

        self.toTensor = Compose([ToImage(), ToDtype(float32, scale = True)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Getter ensures masks are in the correct form.
        imagePath, maskPath = self.dataset[index]
        image = np.array(Image.open(imagePath).convert('RGB'))
        mask = self.remap[np.load(maskPath)]
        result = self.transformCompose(image = image, mask = mask)
        transformedImage = result['image']
        transformedMask = result['mask']

        imageTensor = self.toTensor(transformedImage)
        maskTensor = from_numpy(transformedMask).long()
        return imageTensor, maskTensor
