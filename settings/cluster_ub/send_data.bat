# Send data to cluster
pscp -r %userprofile%\Data\Skin rc621381@ssh-ccub.u-bourgogne.fr:/work/le2i/rc621381/Data/
# Receive Results from cluster
pscp -r rc621381@ssh-ccub.u-bourgogne.fr:/work/le2i/rc621381/Results %userprofile%