#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.amp.autocast_mode
import torch.backends.cuda
import torch.backends.cudnn
import torchvision.transforms.functional as TVF
from danbooru_metadata import TagMappings
from huggingface_hub import hf_hub_download
from Models import VisionModel
from onnxruntime import InferenceSession
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score


BATCH_SIZE = 128
NUM_WORKERS = 8
MINIMUM_TAGS = 16  # Minimum number of times a tag must appear in the dataset to be considered
SMILINGWOLF_THRESHOLD = 0.3537
JOYTAG_THRESHOLD = 0.4
EPS = 1e-8
JOYTAG_MODEL_PATH = Path("../../trainer/cloud-checkpoints/io1nspv6/latest")
METADATA_PATH = Path("../../metadata")


def main():
	# Load posts
	posts = load_posts()

	# Danbooru tag mappings
	tag_mappings = TagMappings(METADATA_PATH)

	# Load models
	sw_model, sw_model_tags = load_smilingwolf(tag_mappings)
	jt_model, jt_model_tags = load_joytag()

	# Intersection between JoyTag and SmilingWolf tags
	intersected_tags = list(set(sw_model_tags).intersection(set(jt_model_tags)))
	print(f"Models have {len(intersected_tags)} tags in common")

	# Parse post tag strings, apply tag mappings, and compute tag counts
	tag_counts = defaultdict(int)

	for post in posts:
		post_tags = set(post['tag_string'].split())
		post_tags = set(tag_mappings.get_canonical(tag) for tag in post_tags)

		for tag in list(post_tags):
			post_tags.update(tag_mappings.get_implications(tag))
		
		post['tags'] = post_tags

		for tag in post['tags']:
			if tag in intersected_tags:
				tag_counts[tag] += 1
	
	# Exclude tags that appear in less than 10 posts
	intersected_tags = [tag for tag in intersected_tags if tag_counts[tag] >= 10]
	print(f"Models have {len(intersected_tags)} tags in common after filtering by usage")
	
	groundtruth_list: list[torch.Tensor] = []

	for post in posts:
		correct_tags = [t in post['tags'] for t in intersected_tags]
		correct_tags = torch.tensor(correct_tags, dtype=torch.bool)
		groundtruth_list.append(correct_tags)

	ground_truth_tensor = torch.stack(groundtruth_list)

	# Compute predictions
	sw_predictions = run_smilingwolf(sw_model, sw_model_tags, posts, intersected_tags)
	del sw_model
	jt_predictions = run_joytag(jt_model, jt_model_tags, posts, intersected_tags)

	# Compute metrics
	sw_metrics = calculate_metrics(sw_predictions, ground_truth_tensor, SMILINGWOLF_THRESHOLD)
	jt_metrics = calculate_metrics(jt_predictions, ground_truth_tensor, JOYTAG_THRESHOLD)

	# Print table of global results
	sw_metrics_mean = {key: np.mean(value) for key, value in sw_metrics.items()}
	jt_metrics_mean = {key: np.mean(value) for key, value in jt_metrics.items()}

	print( "| Model       | Precision | Recall  | F1     | AP     |")
	print( "|-------------|-----------|---------|--------|--------|")
	print(f"| SmilingWolf | {sw_metrics_mean['precision']:.4f}    | {sw_metrics_mean['recall']:.4f}  | {sw_metrics_mean['f1']:.4f} | {sw_metrics_mean['ap']:.4f} |")
	print(f"| JoyTag      | {jt_metrics_mean['precision']:.4f}    | {jt_metrics_mean['recall']:.4f}  | {jt_metrics_mean['f1']:.4f} | {jt_metrics_mean['ap']:.4f} |")

	# Save
	torch.save(sw_predictions, 'sw_predictions.pt')
	torch.save(jt_predictions, 'jt_predictions.pt')
	torch.save(ground_truth_tensor, 'ground_truth.pt')

	with open('validation_post_ids.txt', 'w') as f:
		for post in posts:
			f.write(f"{post['id']}\n")
	
	with open('detailed_metrics.txt', 'w') as f:
		for i, tag in enumerate(intersected_tags):
			f.write(f"{tag}\n")
			f.write(f"{'SmilingWolf':<20}: Precision {sw_metrics['precision'][i]:.4f}, Recall {sw_metrics['recall'][i]:.4f}, F1 {sw_metrics['f1'][i]:.4f}, AP {sw_metrics['ap'][i]:.4f}\n")
			f.write(f"{'JoyTag':<20}: Precision {jt_metrics['precision'][i]:.4f}, Recall {jt_metrics['recall'][i]:.4f}, F1 {jt_metrics['f1'][i]:.4f}, AP {jt_metrics['ap'][i]:.4f}\n")
			f.write('\n')


def calculate_metrics(predictions: torch.Tensor, ground_truth: torch.Tensor, treshold: float) -> dict[str, np.ndarray]:
	"""
	Calculates precision, recall, and F1 score for a set of predictions.
	predictions: A float32 tensor of shape (N, T) where N is the number of posts and T is the number of tags.
	ground_truth: A bool tensor of shape (N, T) where N is the number of posts and T is the number of tags.
	"""
	thresholded = predictions >= treshold
	tp = (thresholded & ground_truth).sum(dim=0)
	fp = (thresholded & ~ground_truth).sum(dim=0)
	#tn = (~thresholded & ~ground_truth).sum(dim=0)
	fn = (~thresholded & ground_truth).sum(dim=0)
	precision = tp / (tp + fp + EPS)
	recall = tp / (tp + fn + EPS)
	f1 = 2 * precision * recall / (precision + recall + EPS)
	ap = np.fromiter((average_precision_score(ground_truth[:, i].numpy(), predictions[:, i].numpy()) for i in range(predictions.shape[1])), dtype=np.float32)

	return {
		'precision': precision.numpy(),
		'recall': recall.numpy(),
		'f1': f1.numpy(),
		'ap': ap,
	}


def load_smilingwolf(tag_mappings: TagMappings) -> tuple[InferenceSession, list[str]]:
	tag_path = Path(hf_hub_download(repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2', revision='v2.0', filename="selected_tags.csv"))

	model = InferenceSession('wd-v1-4-vit-tagger-v2.onnx', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
	model_tags = pd.read_csv(tag_path)
	model_tags = list(model_tags['name'])

	# Map SmilingWolf tags, just in case
	for i in range(len(model_tags)):
		tag = model_tags[i]
		canonical = tag_mappings.get_canonical(tag)

		if tag != canonical:
			print(f"WARNING: Mapping SW Tag: {tag} -> {canonical}")
			model_tags[i] = canonical

	return model, model_tags


def read_top_tags(model_path: Path) -> list[str]:
	with open(model_path / 'top_tags.txt', 'r') as f:
		tags = f.read().splitlines()
	tags = [tag.strip() for tag in tags if tag.strip() != '']
	return tags


def load_joytag() -> tuple[VisionModel, list[str]]:
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	joytag_tags = read_top_tags(JOYTAG_MODEL_PATH)
	joytag_model = VisionModel.load_model(JOYTAG_MODEL_PATH, device='cuda')

	return joytag_model, joytag_tags


def post_to_path(post: dict) -> Path:
	return Path('resized') / f"{post['id']}.webp"


def load_posts() -> list[dict]:
	with open('posts.jsonl', 'r') as f:
		posts = [json.loads(line) for line in f]
	
	posts = [post for post in posts if post_to_path(post).exists()]

	posts = sorted(posts, key=lambda post: post['id'])
	posts = posts[:2**15]

	print(f"Loaded {len(posts)} posts, min id {posts[0]['id']}, max id {posts[-1]['id']}")

	return posts


def make_square(img: np.ndarray, target_size: int) -> np.ndarray:
	old_size = img.shape[:2]
	desired_size = max(old_size)
	desired_size = max(desired_size, target_size)

	delta_w = desired_size - old_size[1]
	delta_h = desired_size - old_size[0]
	top, bottom = delta_h // 2, delta_h - (delta_h // 2)
	left, right = delta_w // 2, delta_w - (delta_w // 2)

	color = [255, 255, 255]
	new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	return new_im


def smart_resize(img: np.ndarray, size: int) -> np.ndarray:
	# Assumes the image has already gone through make_square
	if img.shape[0] > size:
		img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
	elif img.shape[0] < size:
		img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
	return img


def run_smilingwolf(model: InferenceSession, model_tags: list[str], posts: list[dict], intersected_tags: list[str]) -> torch.Tensor:
	tag_indices = np.asarray([model_tags.index(t) for t in intersected_tags], dtype=np.int64)

	_, height, _, _ = model.get_inputs()[0].shape
	dataset = ImageDataset(posts, height, 'sw')
	dl = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False, shuffle=False)

	predictions_list: list[torch.Tensor] = []
	input_name = model.get_inputs()[0].name
	label_name = model.get_outputs()[0].name

	pbar = tqdm(total=len(posts), dynamic_ncols=True, desc="SmilingWolf predictions")
	for batch in dl:
		images = batch['image'].numpy()
		confidents = model.run([label_name], {input_name: images})[0]
		assert len(confidents.shape) == 2 and confidents.shape[0] == len(images) and confidents.shape[1] == len(model_tags) and confidents.dtype == np.float32

		preds = torch.tensor(confidents[:, tag_indices], dtype=torch.float32)

		for i in range(len(preds)):
			predictions_list.append(preds[i])
		
		pbar.update(len(images))
	
	pbar.close()
	
	return torch.stack(predictions_list)


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
	# Pad image to square
	image_shape = image.size
	max_dim = max(image_shape)
	pad_left = (max_dim - image_shape[0]) // 2
	pad_top = (max_dim - image_shape[1]) // 2

	padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
	padded_image.paste(image, (pad_left, pad_top))

	# Resize image
	if max_dim != target_size:
		padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
	
	# Convert to tensor
	image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

	# Normalize
	image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

	return image_tensor


@torch.no_grad()
def run_joytag(model: VisionModel, model_tags: list[str], posts: list[dict], intersected_tags: list[str]) -> torch.Tensor:
	tag_indices = torch.tensor([model_tags.index(t) for t in intersected_tags], dtype=torch.long, device='cpu')

	dataset = ImageDataset(posts, model.image_size, 'jt')
	dl = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False, shuffle=False)

	model.eval()
	joytag_predictions_list: list[torch.Tensor] = []

	pbar = tqdm(total=len(posts), dynamic_ncols=True, desc="JoyTag predictions")
	for batch in dl:
		batch['image'] = batch['image'].to('cuda')

		# Forward
		with torch.amp.autocast_mode.autocast('cuda', enabled=True):
			preds = model(batch)
		
		predictions = torch.sigmoid(preds['tags'].to(torch.float32)).detach().cpu()
		assert len(predictions.shape) == 2 and predictions.shape[0] == len(batch['image']) and predictions.shape[1] == len(model_tags) and predictions.dtype == torch.float32

		predictions_intersected = predictions[:, tag_indices]

		for i in range(len(predictions)):
			#predictions_dict = dict(zip(model_tags, predictions[i]))
			#predictions_intersected = torch.tensor([predictions_dict[t].item() for t in intersected_tags], dtype=torch.float32)
			joytag_predictions_list.append(predictions_intersected[i])
		
		pbar.update(len(batch['image']))

	return torch.stack(joytag_predictions_list)


class ImageDataset(Dataset):
	def __init__(self, posts: list[dict], target_size: int, variant: str):
		self.posts = posts
		self.target_size = target_size
		self.variant = variant
	
	def __len__(self):
		return len(self.posts)
	
	def __getitem__(self, index: int):
		post = self.posts[index]
		path = post_to_path(post)

		if self.variant == 'sw':
			image = load_sw_image(path, self.target_size)
		elif self.variant == 'jt':
			image = Image.open(path)
			image = prepare_image(image, self.target_size)
		else:
			raise ValueError(f"Unknown variant {self.variant}")

		return {
			'image': image,
		}


def load_sw_image(path: Path, target_size: int) -> np.ndarray:
	image = Image.open(path)

	image = image.convert('RGB')
	image = np.asarray(image)

	image = image[:, :, ::-1]

	image = make_square(image, target_size)
	image = smart_resize(image, target_size)
	image = image.astype(np.float32)

	return image


if __name__ == '__main__':
	main()