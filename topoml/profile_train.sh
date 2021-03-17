#!/bin/bash
conda init bash
#source ~/.bashrc
conda activate topoml

cd /home/sam/Documents/PhD/Research/GeoMSCxML/topoml

pwd

filename = $1".log"

mprof run TrainMSCGNN.py 2>&1 ./$filename
mprof plot
