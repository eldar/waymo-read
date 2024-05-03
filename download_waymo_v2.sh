#!/bin/bash

# NOTE: Before proceeding, you need to fill out the Waymo terms of use and complete `gcloud auth login`.

source=gs://waymo_open_dataset_v_2_0_1/training
dest=/home/eldar/work/data/datasets/waymo

tags=(\
stats \
camera_image \
camera_box \
camera_hkp \
camera_to_lidar_box_association \
lidar \
lidar_box \
)

# tags=(\
# camera_box \
# lidar_box \
# )

seqs=(\
10023947602400723454_1120_000_1140_000 \
7670103006580549715_360_000_380_000 \
)

for ((i = 0; i < ${#tags[@]}; i++)); do
    mkdir -p ${dest}/${tags[i]}
    for ((j = 0; j < ${#seqs[@]}; j++)); do
        gsutil cp -n ${source}/${tags[i]}/${seqs[j]}.parquet ${dest}/${tags[i]}/
    done
done

# codec file for ray decompression
mkdir -p ${dest}/veh_asset_ray_compressed/
mkdir -p ${dest}/ped_asset_ray_compressed/
gsutil cp -n ${source}/veh_asset_ray_compressed/codec_config.json ${dest}/veh_asset_ray_compressed/
gsutil cp -n ${source}/ped_asset_ray_compressed/codec_config.json ${dest}/ped_asset_ray_compressed/
