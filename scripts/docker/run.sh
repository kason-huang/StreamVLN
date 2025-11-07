#!/bin/bash
export DATA_PATH=/mnt/sfs_turbo/data-platform/streamvln_data
export CKPT_PATH=/mnt/sfs_turbo/data-platform/streamvln_ckpt

docker compose -f ./scripts/docker/docker-compose.yml up -d