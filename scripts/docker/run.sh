#!/bin/bash
export DATA_PATH=/mnt/sfs_turbo/data-platform/streamvln_data
export CKPT_PATH=/mnt/sfs_turbo/data-platform/streamvln_ckpt

docker compose -f .docker/docker-compose.yml -f .docker/docker-compose.streamvln.yml up -d