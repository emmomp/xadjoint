# xadjoint
Python scripts for reading and analysing adjoint sensitivities from MITgcm.

Intended for reading binary files produced by ECCOv4 adjoint runs of the form .data/.meta
Can read both ADJ and adxx type files.
Method for writing to nctiles format.
At the moment provides an experiment class to define an adjoint experiment, then read/write data from sensitivity files.

GET STARTED: See examples.py for example of how to create an experiment and call related functions.

Relys on xarray, xmitgcm and ecco-v4-python (https://ecco-v4-python-tutorial.readthedocs.io/)

For more information on the ECCO project see 
https://eccov4.readthedocs.io/en/latest/
