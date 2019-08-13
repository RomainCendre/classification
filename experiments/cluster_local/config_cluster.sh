# Create working directory
mkdir ~/Data
mkdir ~/Data/Skin/
mkdir ~/Data/Skin/Saint_Etienne/
mkdir ~/envs
mkdir ~/Results

# Get our project
git clone https://github.com/RomainCendre/classification.git

# Create environment
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
~/anaconda3/bin/conda env create -f ~/classification/toolbox/environment/environment.yml


