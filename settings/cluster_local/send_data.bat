# Send data to cluster
pscp -r %userprofile%\Data\Skin rcendre@calcul.6306-1.rp.u-bourgogne.fr:/home/rcendre/Data/
pscp -r %userprofile%\Data\Neck rcendre@calcul.6306-1.rp.u-bourgogne.fr:/home/rcendre/Data/
# Receive Results from cluster
pscp -r rcendre@calcul.6306-1.rp.u-bourgogne.fr:/home/rcendre/Results %userprofile%