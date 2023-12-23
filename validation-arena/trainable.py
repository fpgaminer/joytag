#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import multiprocessing


MAX_IMAGE_SIZE = 1024
WORKERS = 8


def main():
	with open('posts.jsonl', 'r') as f:
		posts = [json.loads(line) for line in f]

	with multiprocessing.Pool(WORKERS) as pool:
		for _ in tqdm(pool.imap_unordered(handle_post, posts), total=len(posts)):
			pass


def handle_post(post: dict):
	source_path = Path('originals') / str(post['id'])
	target_path = Path('resized') / f"{post['id']}.webp"
	tmp_path = Path('resized') / f"{post['id']}.tmp"

	if target_path.exists():
		return
	
	try:
		img = Image.open(source_path)
		img.load()
	except:
		return
	
	# Check if gif
	if img.format == 'GIF' or (hasattr(img, 'is_animated') and img.is_animated):
		return
	
	# Check if has transparency
	if has_transparency(img):
		return
	
	# Convert to RGB
	try:
		img = img.convert('RGB')
	except:
		return
	
	# Resize
	if max(img.size) > MAX_IMAGE_SIZE:
		scale = MAX_IMAGE_SIZE / max(img.size)
		img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.BICUBIC)
	
	new_img = Image.new('RGB', img.size, (255, 255, 255))
	new_img.paste(img)

	# Save
	tmp_path.parent.mkdir(parents=True, exist_ok=True)
	new_img.save(tmp_path, 'WEBP', quality=80)
	tmp_path.rename(target_path)


# Label images that can't be opened
def has_transparency(img):
	if img.info.get('transparency', None) is not None:
		return True
	if img.mode == 'P':
		transparent = img.info.get('transparency', -1)
		for _, index in img.getcolors():
			if index == transparent:
				return True
	elif img.mode == 'RGBA':
		extrema = img.getextrema()
		if extrema[3][0] < 255:
			return True
	
	return False


if __name__ == '__main__':
	main()