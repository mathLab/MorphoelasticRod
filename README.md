# MorphoelasticRod

In this repository you find two standalone Python codes that implement the 3D morphoelastic rod models for plant shoots described in the preprint "Nutations in plant shoots: Endogenous and exogenous factors in the presence of mechanical deformations" by Agostinelli D., DeSimone A. and Noselli G. (2020). 
The FEniCS Project Version 2019.1.0 is required.

1. ConstantShoot.py is the reduced model in which the rod has spontaneous strains that evolve in time while the length of the shoot is constant. 

2. GrowingShoot.py models the evolution of the shoot while accounting for tip growth, which is assumed constant along the whole elongation zone.

These codes can be used to obtain the results shown in the abovementioned preprint. They generate a sequence of images of the 3D rod configuration (see function save_plot) and of its tip projection on the (e1,e3)-plane (see function save_plot2), which are saved in the local subfolders "Movie" and "Movie2", respectively. Computed solutions are saved in the local subfolder "Data" by means of the function save_data. Please, comment out the calls to these functions in order to suppress the undesired output.

Model parameters are all gathered in the "MATERIAL PARAMETERS" section and some specific functions (e.g., to introduce perturbating apical loads) are defined in the "FEM IMPLEMENTATION" section.

If you use these codes, we would be grateful if you would cite our publication.
