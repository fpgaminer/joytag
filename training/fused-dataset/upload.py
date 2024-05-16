#!/usr/bin/env python3
from datasets import load_dataset, DatasetDict
import datasets


def main():
	with open('top_tags.txt', 'r') as f:
		top_tags = [line.strip() for line in f.readlines() if line.strip()]

	features = datasets.Features({
		'post_id': datasets.Value('int64'),
		'tags': datasets.Sequence(datasets.ClassLabel(names=top_tags)),
		'hash': datasets.Value('binary'),
	})

	print("Loading dataset")
	ds = load_dataset("parquet", data_files={"train": "train.parquet", "validation": "validation.parquet"}, features=features)
	assert isinstance(ds, DatasetDict)

	print(f"Train dataset size: {len(ds['train'])}")
	print(f"Validation dataset size: {len(ds['validation'])}")

	print("Pushing to hub")
	ds.push_to_hub("danbooru2021-slim-v3", private=True)


if __name__ == '__main__':
	main()