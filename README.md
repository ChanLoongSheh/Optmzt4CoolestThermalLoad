# Optimization calculation for the minimum of thermal load across CIE 1931 color space (CIE xy chromaticity)
The problem is defiened clearly in the Nature communications paper from Li, et al "Photonic thermal management of coloured objects". 
But they did not consider the ***photoluminescence (PL)*** process and consequent ***light extraction (LE) effect***.
While Wang, et al. (Science Bulletin, Sub-ambient full-color passive radiative cooling under sunlight based on efficient quantum-dot photoluminescence) and 
Min, et al. (ACS photonics,All-ColorSub-ambient Radiative Cooling Based on Photoluminescence)considered the ***PL***, 
even paper from Science bulletin considered the objective function totoal thermal load of object with ***PL***, 
they did not redistribute the absorbed energy to the emission band to calculate the reflectance.
ACS photonics paper redistributed the energy however ignored the ***LE*** effect

## light extraction induced by energy redistribution
Here we consider energy redistribution as gaussian distribution calculated by adding one more variable cut-off wavelength lambda
