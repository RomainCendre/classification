# Create working directory
mkdir /work/le2i/rc621381/Data
mkdir /work/le2i/rc621381/Features
mkdir /work/le2i/rc621381/Results
mkdir /work/le2i/rc621381/XDG
mkdir /work/le2i/rc621381/conda/envs
# Now create link to these folders
ln -s /work/le2i/rc621381/Data ~/Data
ln -s /work/le2i/rc621381/Features ~/Features
ln -s /work/le2i/rc621381/Results ~/Results
ln -s /work/le2i/rc621381/conda/envs ~/my-envs
ln -s /work/le2i/rc621381/conda/tmp ~/.conda/pkgs

cd /work/le2i/rc621381/
module load pytorch/19.01-nvidia-singularity
singularity-exec-pytorch conda env create -p /work/le2i/rc621381/conda/envs/GPU -f environmentGPU.yml


