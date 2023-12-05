#!/bin/bash
set -euxo pipefail

pip install -r requirements.txt
colossalai run --nproc_per_node 8 timer_trainer_with_cifar10.py --config tpconfig.py
