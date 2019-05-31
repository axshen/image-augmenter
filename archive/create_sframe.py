import numpy as np
import turicreate as tc
import sys
import os

def row_to_bbox_coordinates(row):
    """
    Takes a row and returns a dictionary representing bounding
    box coordinates:  (center_x, center_y, width, height)  e.g. {'x': 100, 'y': 120, 'width': 80, 'height': 120}
    """
    return {'x': row['xmin'] + (row['xmax'] - row['xmin'])/2,
            'width': (row['xmax'] - row['xmin']),
            'y': row['ymin'] + (row['ymax'] - row['ymin'])/2,
            'height': (row['ymax'] - row['ymin'])}

def read_csv_SFrame(csv):
    """
    Read SFrame into appropriate format
    """
    csv_sf = tc.SFrame.read_csv(csv)
    csv_sf['coordinates'] = csv_sf.apply(row_to_bbox_coordinates)
    del csv_sf['xmin'], csv_sf['xmax'], csv_sf['ymin'], csv_sf['ymax']
    csv_sf = csv_sf.rename({'class': 'label', 'filename': 'name'})
    return(csv_sf)

def read_image_SFrame(path):
    """
    Read images from directory into SFrame file
    """
    sf_images = tc.image_analysis.load_images(path, recursive = True,
                                              random_order = True)
    info = sf_images['path'].apply(lambda path: os.path.basename(path).split('/')[:1])
    info = info.unpack().rename({'X.0': 'name'})
    sf_images = sf_images.add_columns(info)
    del sf_images['path']
    return(sf_images)

def create_SFrame(annots, images):
    """
    Join annotations and images in sf format
    """
    sf_annots = read_csv_SFrame(annots)
    sf_images = read_image_SFrame(images)
    sf_annots = sf_annots.pack_columns(['label', 'coordinates'], new_column_name='bbox', dtype=dict)
    sf_annots = sf_annots.groupby('name',
        {'annotations': tc.aggregate.CONCAT('bbox')})
    sf = sf_images.join(sf_annots, on='name', how='inner')
    sf['annotations'] = sf['annotations'].fillna([])
    return(sf)

# paths
gen_annots_path = 'annotations/annotations_updated.csv'
gen_images_path = 'generated/'

raw_annots_path = 'annotations/annotations_cleaned.csv'
raw_images_path = 'images/'

# Read annotations
gen_sf = create_SFrame(gen_annots_path, gen_images_path)
raw_sf = create_SFrame(raw_annots_path, raw_images_path)

print(gen_sf.shape)
print(raw_sf.shape)

sf = raw_sf.append(gen_sf)

print(sf.shape)
sf.save('data_sf/soccer_balls.sframe')
