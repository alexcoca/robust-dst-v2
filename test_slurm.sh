#!/bin/bash
#SBATCH -J dummy-test
#SBATCH --time=00:01:30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --chdir=/rds/project/rds-DuWT62BKvk8/ac2123/robust-dst-v2
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

echo "=== DUMMY TEST START ==="
echo "Hostname: $(hostname)"
echo "Date:     $(date)"
echo "PWD:      $(pwd)"
sleep 10
echo "=== DUMMY TEST END ==="
