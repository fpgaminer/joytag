import torch.nn as nn
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TVF
from torchvision.transforms.autoaugment import _apply_op


# I re-implement the TrivialAugmentWide class from the torchvision.transforms module
# So that I can tweak the augmentations used
class TrivialAugmentMod(nn.Module):
	def __init__(self, level: int):
		super().__init__()
		self.num_magnitude_bins = 31
		self.interpolation = InterpolationMode.BILINEAR
		self.fill = [255., 255., 255.]
		self.wide = True
		self.level = level
	
	def _augmentation_space(self, num_bins: int) -> dict[str, tuple[torch.Tensor, bool]]:
		space = {
			'Identity': (torch.tensor(0.0), False),
			'ShearX': (torch.linspace(0.0, 0.99, num_bins), True),
			'ShearY': (torch.linspace(0.0, 0.99, num_bins), True),
			'Rotate': (torch.linspace(0.0, 135.0, num_bins), True),
			#'Color': (torch.linspace(0.0, 0.99, num_bins), True),
			#'Contrast': (torch.linspace(0.0, 0.99, num_bins), True),
			'Brightness': (torch.linspace(0.0, 0.99, num_bins), True),
			'Sharpness': (torch.linspace(0.0, 0.99, num_bins), True),
			#'Posterize': (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
			#'Solarize': (torch.linspace(255.0, 0.0, num_bins), False),
			#'AutoContrast': (torch.tensor(0.0), False),
			#'Equalize': (torch.tensor(0.0), False),
			'TranslateX': (torch.linspace(0.0, 32.0, num_bins), True),
			'TranslateY': (torch.linspace(0.0, 32.0, num_bins), True),
		}

		if self.level == 2:
			space = space | {
				'Color': (torch.linspace(0.0, 0.99, num_bins), True),
				'Contrast': (torch.linspace(0.0, 0.99, num_bins), True),
				'Posterize': (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
				'Solarize': (torch.linspace(255.0, 0.0, num_bins), False),
				'AutoContrast': (torch.tensor(0.0), False),
				'Equalize': (torch.tensor(0.0), False),
			}
		
		return space
	
	def forward(self, x):
		fill = self.fill
		channels, height, width = TVF.get_dimensions(x)

		op_meta = self._augmentation_space(self.num_magnitude_bins)
		op_index = int(torch.randint(len(op_meta), (1,)).item())
		op_name = list(op_meta.keys())[op_index]
		magnitudes, signed = op_meta[op_name]
		magnitude = (
			float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
			if magnitudes.ndim > 0
			else 0.0
		)
		if signed and torch.randint(2, (1,)):
			magnitude *= -1.0
		
		return _apply_op(x, op_name, magnitude, interpolation=self.interpolation, fill=fill)


# Adapted from (https://github.com/SmilingWolf/SW-CV-ModelZoo/tree/main)
class SWAugment(nn.Module):
	def __init__(self, noise_level: int, image_size: int = 448):
		super().__init__()
		self.noise_level = noise_level
		self.image_size = image_size
	
	def random_flip(self, x: Image.Image) -> Image.Image:
		"""Random horizontal flip"""
		if torch.rand(1) < 0.5:
			return x.transpose(Image.FLIP_LEFT_RIGHT)
		return x
	
	def random_crop(self, x: Image.Image) -> Image.Image:
		"""Assumes the image is square"""
		image_shape = x.size

		# factor between 0.87 and 0.998
		factor = 0.87 + torch.rand(1) * 0.128

		new_size = min(image_shape) * factor
		new_size = int(new_size.item())

		offset_x = int(torch.randint(0, image_shape[0] - new_size + 1, size=(1,)).item())
		offset_y = int(torch.randint(0, image_shape[1] - new_size + 1, size=(1,)).item())

		x = x.crop((offset_x, offset_y, offset_x + new_size, offset_y + new_size))
		assert x.size == (new_size, new_size)

		return x

	def resize(self, x: Image.Image) -> Image.Image:
		interpolation_methods = [
			Image.NEAREST,
			Image.BILINEAR,
			Image.BICUBIC,
		]
		interpolation_method = interpolation_methods[int(torch.randint(0, 3, size=(1,)).item())]

		return x.resize((self.image_size, self.image_size), interpolation_method)
	
	def make_square(self, x: Image.Image) -> Image.Image:
		"""Pad the image to make it square"""
		width, height = x.size
		max_dim = max(width, height)
		pad_left = (max_dim - width) // 2
		pad_top = (max_dim - height) // 2

		new_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
		new_image.paste(x, (pad_left, pad_top))

		return new_image
	
	def random_rotate(self, x: Image.Image) -> Image.Image:
		"""Random rotation between -45 and 45, fill value is 255"""
		interpolation_methods = [
			Image.NEAREST,
			Image.BILINEAR,
			Image.BICUBIC,
		]

		interpolation_method = interpolation_methods[int(torch.randint(0, 3, size=(1,)).item())]
		angle = (torch.rand(1) * 90 - 45).item()

		x = x.rotate(angle, interpolation_method, fillcolor=(255, 255, 255))
		assert x.size == (self.image_size, self.image_size)

		return x
	
	def forward(self, x: Image.Image) -> Image.Image:
		# Pad to square
		x = self.make_square(x)

		if self.noise_level >= 1:
			x = self.random_flip(x)
			x = self.random_crop(x)

		# Resize to the target image size		
		x = self.resize(x)

		if self.noise_level >= 1:
			x = self.random_rotate(x)
		
		return x


class SWMixUp(nn.Module):
	def __init__(self, mixup_alpha: float):
		super().__init__()
		self.mixup_alpha = mixup_alpha
	
	def forward(self, batch):
		batch_size = batch['image'].shape[0]

		x = batch['image']
		y = batch['tags']

		images_two = x.flip(0)
		labels_two = y.flip(0)

		l = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample((int(batch_size), 1, 1, 1)).to(x.device)
		x_l = l.reshape(batch_size, 1, 1, 1)
		y_l = l.reshape(batch_size, 1)

		x = x * x_l + images_two * (1 - x_l)
		y = y * y_l + labels_two * (1 - y_l)

		return batch | {'image': x, 'tags': y}