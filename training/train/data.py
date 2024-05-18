from typing import Optional, Tuple
from torch.utils.data import Dataset
import torch
import torch.utils.data
from data_augmentation import SWAugment, SWMixUp, TrivialAugmentMod
from PIL import Image
from pathlib import Path
from torchvision.transforms import functional as TF
from torch.utils.data.distributed import DistributedSampler
import io


class ImageDataset(Dataset):
	def __init__(
		self,
		source,
		n_tags: int,
		images_path: Path,
		data_augmentation: str | None = None,
		label_smoothing: float = 0.0,
		include_hash: bool = False,
	):
		"""
		Args:
			source: Source dataset
			n_tags: Number of different tags
			images_path: Path to the images
			data_augmentation: Data augmentation type (One of: 'trivial', 'sw', None)
			label_smoothing: Label smoothing factor
			include_hash: Include hash in the output (for debugging purposes)
		"""
		self.source = source
		self.n_tags = n_tags
		self.data_augmentation = data_augmentation
		self.label_smoothing = label_smoothing
		self.include_hash = include_hash
		self.images_path = Path(images_path)

		if self.data_augmentation == 'trivial':
			self.augment_layer = TrivialAugmentMod(level=1)
		elif self.data_augmentation == 'trivial2':
			self.augment_layer = TrivialAugmentMod(level=2)
		elif self.data_augmentation == 'sw':
			self.augment_layer = SWAugment(2)
		elif self.data_augmentation is None:
			self.augment_layer = None
		else:
			raise ValueError(f"Unknown data augmentation type: {self.data_augmentation}")
	
	def __len__(self):
		return len(self.source)
	
	def __getitem__(self, key: Tuple[int, int]) -> dict:
		index, target_image_size = key
		row = self.source[index]

		# Load image
		if 'image' in row:
			image = Image.open(io.BytesIO(row['image']))
		else:
			hash = row['hash'].hex()
			image = Image.open(self.images_path / hash[:2] / hash[2:4] / f"{hash}.webp")
		
		if self.data_augmentation != 'sw':
			image_size = max(image.size)

			# Pad image to square
			square_image = Image.new('RGB', (image_size, image_size), (255, 255, 255))
			square_image.paste(image, ((image_size - image.size[0]) // 2, (image_size - image.size[1]) // 2))
			image = square_image

			# Resize image to target size
			if image.size != (target_image_size, target_image_size):
				scale = target_image_size / max(image.size)
				image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)), Image.BILINEAR)# Image.BICUBIC)
				assert max(image.size) == target_image_size
		
		if isinstance(self.augment_layer, SWAugment):
			self.augment_layer.image_size = target_image_size

		# Apply data augmentation
		if self.augment_layer is not None:
			image = self.augment_layer(image)
		
		# Convert to Tensor [0-255]
		# Normalization, etc is performed on the GPU
		image = TF.pil_to_tensor(image)
		
		# Convert row['tags'] from a list of ints to multi-hot vector
		if self.label_smoothing > 0.0:
			tags = torch.zeros(self.n_tags, dtype=torch.float32)
			tags.fill_(self.label_smoothing)
			tags.scatter_(0, row['tags'], 1.0 - self.label_smoothing)
		else:
			tags = torch.zeros(self.n_tags, dtype=torch.float32).scatter_(0, row['tags'], 1.0)

		result = {
			# x
			'image': image,

			# y
			'tags': tags,
		}

		if self.include_hash:
			result['hash'] = row['hash']
		
		return result


class GPUDataProcessing(torch.nn.Module):
	"""
	Normalizes the image and applies any post batching data augmentation like mixup.
	"""
	def __init__(self, mixup_alpha: float = 0.0):
		super().__init__()
		self.mixup_layer = SWMixUp(mixup_alpha) if mixup_alpha > 0.0 else None
	
	def forward(self, batch):
		if 'image' not in batch:
			return batch
		
		# Put into range [0-1]
		batch['image'] = batch['image'] / 255.0

		# Mixup
		if self.mixup_layer is not None:
			batch = self.mixup_layer(batch)
		
		# Normalize
		batch['image'] = TF.normalize(batch['image'], mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
		
		return batch


class BetterDistributedSampler(DistributedSampler):
	def __init__(
		self,
		dataset: Dataset,
		target_image_size: int,
		num_replicas: Optional[int] = None,
		rank: Optional[int] = None,
		shuffle: bool = True,
		seed: int = 0,
		drop_last: bool = False,
	) -> None:
		super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
		self.resume_index = None
		self.target_image_size = target_image_size
	
	def set_state(self, epoch: int, index: int) -> None:
		"""
		Sets the epoch and fast forwards the iterator to the given index.
		Needs to be called before the dataloader is iterated over.
		"""
		self.set_epoch(epoch)
		self.resume_index = index

	def __iter__(self):
		i = super().__iter__()

		# Add resolution to the iterator
		target_image_size = self.target_image_size
		i = ((x, target_image_size) for x in i)

		if self.resume_index is not None:
			for _ in range(self.resume_index):
				next(i)
			self.resume_index = None
		
		return i