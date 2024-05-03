from typing import Optional
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2

import matplotlib.pyplot as plt


# Path to the directory with all components
dataset_dir = '/home/eldar/work/data/datasets/waymo'

context_name = '10023947602400723454_1120_000_1140_000'

def read(tag: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
    return dd.read_parquet(paths)


# Lazily read camera images and boxes 
cam_image_df = read('camera_image')
cam_box_df = read('camera_box')

# Combine DataFrame for individual components into a single DataFrame.

# Camera cam_box_df will be grouped, so each row will have a camera image
# and all associated boxes.
image_w_box_df = v2.merge(cam_image_df, cam_box_df, right_group=True)

# Show raw data
image_w_box_df.head()

# Example how to access data fields via v2 object-oriented API
print(f'Available {image_w_box_df.shape[0].compute()} rows:')
for i, (_, r) in enumerate(image_w_box_df.iterrows()):
    # Create component dataclasses for the raw data
    cam_image = v2.CameraImageComponent.from_dict(r)
    cam_box = v2.CameraBoxComponent.from_dict(r)
    #   print(
    #       f'context_name: {cam_image.key.segment_context_name}'
    #       f' ts: {cam_image.key.frame_timestamp_micros}'
    #       f' camera_name: {cam_image.key.camera_name}'
    #       f' image size: {len(cam_image.image)} bytes.'
    #       f' Has {len(cam_box.key.camera_object_id)} camera labels:'
    #   )
    cam_name = cam_image.key.camera_name
    if cam_name != 1:
        continue
    img = tf.image.decode_jpeg(cam_image.image)
    plt.imshow(img)
    plt.show()
