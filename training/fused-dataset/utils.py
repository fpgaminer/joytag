import sqlite3
from itertools import islice
from pathlib import Path
from typing import TypeVar, Iterator, Iterable


TAG_MACHINE_PATH = Path('/home/night/tag-machine/rust-api/images')


def open_aux_db():
	db = sqlite3.connect('aux.sqlite3')
	cursor = db.cursor()

	# Schema
	with open('schema.sql', 'r') as f:
		cursor.executescript(f.read())
	
	db.commit()
	cursor.close()

	return db


T = TypeVar('T')

def batcher(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
	iterator = iter(iterable)
	while batch := list(islice(iterator, n)):
		yield batch