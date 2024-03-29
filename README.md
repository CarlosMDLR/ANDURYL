
![logoanduryl](https://github.com/CarlosMDLR/ANDURYL/assets/105994653/0e4215d1-d896-4bd7-9806-0f3f59e4bcf2)

ANDURYL (A bayesiaN Decomposition code for Use in photometRY Labors) is a code created to perform photometric decompositions of galaxies using Bayesian statistics.
This code has been made within the framework of a Master's Thesis, and is in a beta state, fully functional, but can be adapted beyond the needs of the Master's project.
## Instalation
To correctly install ANDURYL you simply have to clone this repository, and then run 
the requirements.txt file, which includes all the Python packages necessary for the correct operation of the code:

```
git clone https://github.com/CarlosMDLR/ANDURYL.git
cd ANDURYL
python -m pip install -r requirements.txt
```
:warning: If you have ```arxiv``` installed on your device, be aware that this may cause incompatibilities with the function of the main program that runs triangular diagrams, if you experience failures in this regard, uninstall arxiv or set "no" to the generation of plots in the configuration file :warning:
## Utilization

There is currently a test environment available in which to use the code to fit an example galaxy.
Although for now, being in the beta phase, the following steps must be followed. In the near future
the code will be made even more generic to adapt its use:
- In the gals_imgs directory the fit files of the galaxies that you want to analyze are introduced.
- In the mask_fits directory, enter the fits file of the mask that you want to use for each galaxy.
- In the SDSS_catalog directory there are two other subdirectories in which one includes information
  about the frame from which the example galaxy has been cut and another the SDSS photoField.
- In the FWHM list directory there is a file with the information of the Moffat Point Spread Function for each galaxy.
- AND_images is the directory in which the images of the residuals and triangular plots will be saved, if indicated in the configuration file.
- The configuration file is ```setup.txt```, in which different parameters and code directories are indicated.

Once all the parameters and files have been placed in their proper places the code can be used by simply running the main program ```ANDURYL_main.py```:

```
python ANDURYL_main.py
```
