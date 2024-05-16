CREATE TABLE IF NOT EXISTS images (
	id INTEGER PRIMARY KEY,
	hash BLOB NOT NULL UNIQUE,
	trainable_image INTEGER,
	phash INTEGER
);

CREATE INDEX IF NOT EXISTS images_trainable_image ON images (trainable_image);