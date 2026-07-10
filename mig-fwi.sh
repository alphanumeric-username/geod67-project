#!/bin/sh

CWD=$(pwd)

JOB_NAME=UNET_FWI-prelu
LOGFILE=$JOB_NAME\_$(date "+%Y-%m-%d_%H-%M-%S").txt

CWD=$CWD sbatch  -A geo-inct -p standard --job-name $JOB_NAME -o $LOGFILE mig-fwi.job.sh