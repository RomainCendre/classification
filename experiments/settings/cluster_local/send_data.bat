# Send data to cluster
pscp -r %userprofile%\Data\Skin\Saint_Etienne\Patients rcendre@calcul.6306-1.rp.u-bourgogne.fr:/home/rcendre/Data/Skin/Saint_Etienne/
# Receive Results from cluster
pscp -r rcendre@calcul.6306-1.rp.u-bourgogne.fr:/home/rcendre/Results %userprofile%