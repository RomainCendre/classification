# Create working directory
## Long term storage
mkdir /work/le2i/rc621381/Data
mkdir /work/le2i/rc621381/Data/Skin/
mkdir /work/le2i/rc621381/Data/Skin/Saint_Etienne/

## Short term storage
mkdir /work/le2i/rc621381/.research
mkdir /work/le2i/rc621381/Results
mkdir /work/le2i/rc621381/XDG
mkdir /work/le2i/rc621381/conda/envs

# Now create link to these folders
## Long term storage shortcut
ln -s /work/le2i/rc621381/Data ~/Data

## Short term storage shortcut
ln -s /work/le2i/rc621381/ ~/work
ln -s ~/work/conda/envs ~/envs
ln -s ~/work/.research ~/.research
ln -s ~/work/Results ~/Results
ln -s ~/work/conda/tmp ~/.conda/pkgs

# Get our project
git clone https://github.com/RomainCendre/classification.git

# Create environment
module load pytorch/19.01-nvidia-singularity
singularity-exec-pytorch conda env create -p ~/work/conda/envs/GPU -f ~/work/classification/toolbox/environment/environmentGPU.yml


