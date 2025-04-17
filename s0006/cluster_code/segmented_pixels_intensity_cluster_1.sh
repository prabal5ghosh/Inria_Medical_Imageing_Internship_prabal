#!/bin/bash 
#OAR -q production
#OAR -l host=1/gpu=1
#OAR -l walltime=00:30:00
#OAR -p gpu_count > 0
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err
# display some information about attributed resources

echo "=== Host and GPU Info ==="
hostname 
nvidia-smi 
nvcc --version
 

echo "=== Loading environment ==="
module load conda
module load cuda/11.8
conda activate mri_2025_4


echo "=== Checking PyTorch GPU Availability ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

echo "=== Starting TotalSegmentator Job ==="
cd /home/prghosh/prabal_ghosh/Inria_Medical_Imageing_Internship_prabal/s0006/cluster_code

# Run your script
python segmented_pixels_intensity_cluster_1.py

echo "=== Done ==="
conda deactivate
