# arbk
This repository contains the Python code that generates the figures in the paper "Acceleration and restart for the randomized Bregman-Kaczmarz method".

##### Authors:
- Lionel Tondji  (<tngoupeyou@aimsammi.org>)
- Ion Necoara  (<ion.necoara@upb.ro>)
- Dirk Lorenz    (<d.lorenz@uni-bremen.de>)

Contents
--------

##### Drivers (run these to generate figures):
	example_overdetermined.ipynb                  notebook to generate figure 5 and Table 4
	example_underdeternined.ipynb          	      notebook to generate figure 2, 3, 4 and table 1, 2, 3
	example_CT.ipynb                              notebook to generate figure 6, 7 and Table 5

##### Routines called by the drivers:
	tools.py                      Python packages containing functions like The accelerated (restart) Bregman-Kaczmarz, myphantom, soft_skrinkage.

The myphantom function generates the data for the CT example.
