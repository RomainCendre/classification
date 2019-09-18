conda activate Skin
# Install kernels
conda install jupyter
jupyter notebook --generate-config
# Extension for variable
conda install -c conda-forge jupyter_contrib_nbextensions
# Create server conf directory
mkdir ~/servers
