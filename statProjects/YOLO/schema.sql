-- Main Image Table
CREATE TABLE images (
    image_id SERIAL PRIMARY KEY,
    image_url TEXT UNIQUE,
    file_path TEXT UNIQUE,
    resolution TEXT,
    date_collected TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Image Categorization Table
CREATE TABLE image_category (
    category_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(image_id),
    category_name TEXT NOT NULL,
    noise_level TEXT CHECK (noise_level IN ('low', 'high', 'medium')) DEFAULT NULL
);

-- Annotations Table
CREATE TABLE annotations (
    annotation_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(image_id),
    bounding_box TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES image_category(category_id),
    confidence_score REAL,
    annotation_source TEXT CHECK (annotation_source IN ('manual', 'automated'))
);

-- Dataset Partition Table
CREATE TABLE dataset_partition (
    partition_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(image_id),
    dataset_split TEXT CHECK (dataset_split IN ('train', 'test', 'val')),
    source_type TEXT CHECK (source_type IN ('COCO_standard', 'COCO_construction', 'Generated', 'Web_scraped'))
);
