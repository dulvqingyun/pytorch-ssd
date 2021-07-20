#!/usr/bin/env bash

# nohup  
python ./train_ssd.py --dataset_type voccoco --datasets /home/ubuntu/data/VOCdevkitCOCO/VOCCOCO \
--validation_dataset /home/ubuntu/data/VOCdevkitCOCO/VOCCOCO \
--net  mb1-ssd \
--base_net models/mobilenet_v1_with_relu_69_5.pth \
--num_epochs 200 \
--scheduler cosine \
--lr 0.01 \
--t_max 200 \
--num_workers 8 \
# > ./ssd_mbv1.log 2>&1 &


