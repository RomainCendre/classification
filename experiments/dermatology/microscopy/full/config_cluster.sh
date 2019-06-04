ln -s /work/le2i/rc621381/data/ data
ln -s /work/le2i/rc621381/conda/envs my-envs
ln -s /work/le2i/rc621381/conda/tmp ~/.conda/pkgs

cd /work/le2i/rc621381/
module load pytorch/19.01-nvidia-singularity
singularity-exec-pytorch conda env create -p /work/le2i/rc621381/conda/envs/GPU -f environmentGPU.yml
singularity-exec-pytorch conda activate /work/le2i/rc621381/conda/envs/GPU

./my-envs/GPU/bin/python my_script.py


