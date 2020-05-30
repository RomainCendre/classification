conda activate Skin
export PYTHONPATH=~/classification
jupyter nbextensions_configurator enable --user
taskset -c 0-11 jupyter notebook --no-browser --ip=10.141.13.128
taskset -c 0-11 jupyter notebook --no-browser --ip=10.141.13.130