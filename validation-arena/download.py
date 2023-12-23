#!/usr/bin/env python3
"""
Download recent Danbooru posts and images.
Images are saved to the `originals` directory.
Posts are saved to the `posts.jsonl` file.
"""
import requests
import json
import time
from tqdm import tqdm
from pathlib import Path
import multiprocessing


STARTING_ID = 6800000


def main():
	posts = get_posts()

	with multiprocessing.Pool(2) as pool:
		for _ in tqdm(pool.imap_unordered(download_post, posts.values()), total=len(posts)):
			pass


def download_post(post: dict):
	if 'file_url' not in post:
		return

	download_path = Path('originals') / str(post['id'])
	if download_path.exists():
		return

	download_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = Path('originals') / f"{post['id']}.tmp"

	try:
		response = requests.get(post['file_url'])
		response.raise_for_status()
		data = response.content
	except Exception as e:
		print(post['id'], e)
		return

	with open(tmp_path, 'wb') as f:
		f.write(data)
	
	tmp_path.rename(download_path)


def get_posts() -> dict[int, dict]:
	posts = {}

	if Path('posts.jsonl').exists():
		with open('posts.jsonl', 'r') as f:
			for line in f:
				post = json.loads(line)
				posts[post['id']] = post

	with tqdm(total=2**16) as pbar, open('posts.jsonl', 'a') as f:
		while len(posts) < 2**16:
			if len(posts) == 0:
				url = f"https://danbooru.donmai.us/posts.json?limit=1000&page=b{STARTING_ID}"
			else:
				min_id = min(post['id'] for post in posts.values())
				url = f"https://danbooru.donmai.us/posts.json?limit=1000&page=b{min_id}"

			response = requests.get(url)
			response.raise_for_status()

			result = response.json()

			for post in result:
				assert post['id'] not in posts
				posts[post['id']] = post
				f.write(json.dumps(post) + '\n')
			
			time.sleep(1)
			pbar.update(len(result))
	
	return posts


if __name__ == '__main__':
	main()