-- Image Management

-- Insert a new image
CREATE OR REPLACE FUNCTION save_image_metadata(image_url TEXT, file_path TEXT, resolution TEXT)
RETURNS INTEGER AS $$
DECLARE new_id INTEGER;
BEGIN
    INSERT INTO images (image_url, file_path, resolution) 
    VALUES (image_url, file_path, resolution) 
    RETURNING image_id INTO new_id;
    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- Retrieve image metadata by ID
CREATE OR REPLACE FUNCTION get_image_metadata(image_id INTEGER)
RETURNS TABLE (image_url TEXT, file_path TEXT, resolution TEXT, date_collected TIMESTAMP) AS $$
BEGIN
    RETURN QUERY SELECT image_url, file_path, resolution, date_collected FROM images WHERE image_id = image_id;
END;
$$ LANGUAGE plpgsql;

-- Delete an image
CREATE OR REPLACE FUNCTION delete_image(image_id INTEGER)
RETURNS VOID AS $$
BEGIN
    DELETE FROM images WHERE image_id = image_id;
END;
$$ LANGUAGE plpgsql;

-- Update image resolution or path
CREATE OR REPLACE FUNCTION update_image(image_id INTEGER, new_resolution TEXT, new_file_path TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE images SET resolution = new_resolution, file_path = new_file_path WHERE image_id = image_id;
END;
$$ LANGUAGE plpgsql;


-- Image Categorization

-- Assign categories to an image
CREATE OR REPLACE FUNCTION save_image_category(image_id INTEGER, category_name TEXT, noise_level TEXT)
RETURNS VOID AS $$
BEGIN
    INSERT INTO image_category (image_id, category_name, noise_level) 
    VALUES (image_id, category_name, noise_level);
END;
$$ LANGUAGE plpgsql;

-- Retrieve all categories for an image
CREATE OR REPLACE FUNCTION get_image_categories(image_id INTEGER)
RETURNS TABLE (category_name TEXT, noise_level TEXT) AS $$
BEGIN
    RETURN QUERY SELECT category_name, noise_level FROM image_category WHERE image_id = image_id;
END;
$$ LANGUAGE plpgsql;


-- Annotations

-- Save an annotation (bounding box + category)
CREATE OR REPLACE FUNCTION save_annotation(image_id INTEGER, bounding_box TEXT, category_name TEXT, confidence_score REAL)
RETURNS VOID AS $$
BEGIN
    INSERT INTO annotations (image_id, bounding_box, category_name, confidence_score) 
    VALUES (image_id, bounding_box, category_name, confidence_score);
END;
$$ LANGUAGE plpgsql;

-- Retrieve all annotations for an image
CREATE OR REPLACE FUNCTION get_annotations(image_id INTEGER)
RETURNS TABLE (bounding_box TEXT, category_name TEXT, confidence_score REAL) AS $$
BEGIN
    RETURN QUERY SELECT bounding_box, category_name, confidence_score 
                 FROM annotations WHERE image_id = image_id;
END;
$$ LANGUAGE plpgsql;

-- Delete an annotation
CREATE OR REPLACE FUNCTION delete_annotation(annotation_id INTEGER)
RETURNS VOID AS $$
BEGIN
    DELETE FROM annotations WHERE annotation_id = annotation_id;
END;
$$ LANGUAGE plpgsql;


-- Dataset Partitioning

-- Assign dataset partition (train/test/val)
CREATE OR REPLACE FUNCTION save_dataset_partition(image_id INTEGER, dataset_type TEXT, source_type TEXT)
RETURNS VOID AS $$
BEGIN
    INSERT INTO dataset_partition (image_id, dataset_type, source_type) 
    VALUES (image_id, dataset_type, source_type);
END;
$$ LANGUAGE plpgsql;

-- Retrieve images by partition type
CREATE OR REPLACE FUNCTION get_images_by_partition(dataset_type TEXT)
RETURNS TABLE (image_id INTEGER, source_type TEXT) AS $$
BEGIN
    RETURN QUERY SELECT image_id, source_type FROM dataset_partition WHERE dataset_type = dataset_type;
END;
$$ LANGUAGE plpgsql;


-- Image Storage and Retrieval

-- Get image file path from storage
CREATE OR REPLACE FUNCTION get_image_file_path(image_id INTEGER)
RETURNS TEXT AS $$
DECLARE file_path TEXT;
BEGIN
    SELECT file_path INTO file_path FROM images WHERE image_id = image_id;
    RETURN file_path;
END;
$$ LANGUAGE plpgsql;


