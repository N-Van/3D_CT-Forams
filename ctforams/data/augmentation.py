import monai


class AugmentationWrapper(object):
    def __init__(self, transforms={}):
        super().__init__()
        transforms_array = [transform for transform in transforms.values()]
        self.transforms = monai.transforms.Compose(transforms_array)

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)
