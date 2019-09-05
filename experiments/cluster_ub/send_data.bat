# Send data to cluster
pscp -r C:\Users\Romain\Data\Skin\Saint_Etienne\Patients rc621381@ssh-ccub.u-bourgogne.fr:/archive/le2i/rc621381/Data/Skin/Saint_Etienne/
# Receive Results from cluster
pscp -r rc621381@ssh-ccub.u-bourgogne.fr:/work/le2i/rc621381/Results C:\Users\Romain\