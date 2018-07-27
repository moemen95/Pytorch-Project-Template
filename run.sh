#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
#python3 main.py PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=1
#python3 main.py configs/dqn_exp_0.json
#python main.py configs/dqn_exp_0.json
python main.py configs/erfnet_exp_0.json
