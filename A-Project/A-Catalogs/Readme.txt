Here I will describe the folders and files on the A-Catalogs folder:
* It is possible that some old notebooks wont run because they are searching for a file in a wrong folder. 
  But every file will be in the ERDA version of the project


Folders:

- Backup : Backups and previous versions of the main catalog.
- Charlotte : Catalogs of the original Binospec sample before I started this project
- Data\large_files : Here are all the data files used in the project. Some names are self explanatory, but in order:
    - completeness: Saved results for completeness testing of Lyman Alpha observations.
    - Fresco: Old version of Fresco slitless NIRCam data
    - Fresco Database: Newest version (at 2024 19 sept) of the FRESCO data of goods North, and footprints
    - FrescoGrisms: F444W fresco data and footprints
    - infall: IGM model and SED models used to study galaxy infall (not relevant for this project)
    - LowLim: Lower limits of escape fraction for Lya only sources
        - old: Old version of LowLim
    - plots: All plots for the Lya profiles, now all plots are in E-plots
    - profiles : very early plots of Lya profiles
    - Raw: Raw data, this should be better downloaded from ERDA
    - Reduced_data: Science results, spectra that should be used to do any Science
    - Star Models: Models of stars used for telluric callibrations
- ForMaster: Some .npy used to create the final catalog in C-FrescoFescVoff
- Romain: New extractions done by Roman from the Geneva Observatory of some of the sources that were contaminated
- Telluric: Output files of telluric callibrations
- test : Some results of fitting power laws to JADES restframe UV data with HST and/or with JWST
- velocity_offsets : Files needed to plot the velocity offset figures, including literature and Charlottes halo mass models
- measurements : Many of the outputs across the Code are saved here, most are used for later writing the master Catalog. 

Files :

- Binospec-Candels.cat : Main Catalog that will be published, all results come from here.
- Bouwens_cat.dat : Catalog from Bouwens works, used to compare any selection bias in our sample through the Hband
- CandelsPhotometry.txt : Candels/SHARDS photometric catalog
- Fresco_* : Versions of the Fresco catalogs
- goodsn_3dhst* : Data from 3DHST photometry used in the project
- Ha_fresco_GN : Data from FRESCOs Halpha paper
- HaLines.cat: Our Lyman Alpha galaxies that had Halpha, and the wavelength that i manually measured (later refined)
- HaLinesNonDetect.cat: Our no-Lyman Alpha galaxies that had Halpha, and the wavelength that i manually measured (later refined)
- *barro19.fits: Barro19 photometric catalog of GOODS-New

