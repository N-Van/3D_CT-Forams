import monai
import monai.transforms


class AugmentationWrapper(object):
    def __init__(self, transforms={}, target="compose"):
        super().__init__()
        transforms_array = [t for t in transforms.values()]
        self.transforms = self.get_type(target)(transforms_array)

    def get_type(self, str_val):
        if str_val == "compose":
            return monai.transforms.Compose
        elif str_val == "oneof":
            return monai.transforms.OneOf
        else:
            raise ValueError(f"Unknown composition type {str_val}")

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)
