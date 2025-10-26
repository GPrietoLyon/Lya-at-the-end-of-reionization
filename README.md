# Binospec data project
​
This Github respository is structured just as an example and can be changed to suit your working practice, but the overall goal is to make it easy to share progress and for me to help debug any problems etc. Also it should help keep results and code organised, as well as encouraging good documentation for yourself.

This repository includes the following directories:
- `data`
- `data/large_files` - Use for large files (>10-100Mb) that can be easily re-downloaded (such as the primary datasets below). GitHub is not designed for large file storage, so files that are in this folder will be ignored for uploading per the `.gitignore` file in the main folder.
- `code`
- `plots`

## Python package objectives

  - Make a [pip installable python package with documentation](https://nsls-ii.github.io/scientific-python-cookiecutter/index.html) to enable you to easily explore Binospec data, that you can later use to search for Lyman alpha emitters :)
  - Always try not to reinvent the wheel too much - `astropy` and `numpy` have lots of useful functions for dealing with fits and array data
  - Package functions:
    * load in a Binspec fits file (**mask class?**)
    * load a single slit (**slit class?**) and make the following plots:
      + 1D spectrum with option to pick the Y position + Y aperture size
      + 2D spectrum with options to pick the wavelength, Y positions and widths
      + add a marker (circle/arrow etc) to the expected position of the science target
      + ...

This package doesn't need to be hosted on PyPI for now, i.e. you don't need to add it to do [these steps] (https://nsls-ii.github.io/scientific-python-cookiecutter/publishing-releases.html) but it should be possible to pip install it on your own computer and for other people to download it from your github repo and install it. E.g. similar to [this](https://github.com/charlottenosam/kmos_tools) package

As the project progresses, you can break these down into more detailed steps and update or add new steps as required. You could use github issues and milestones to track your progress and to-do lists.

I recommend creating the package in a new github repository (in this organization) so that it is separate from the data (i.e. so you can use the package on any new data). You can use this repository for exploring the data and testing your package.

## Data

The data folder contains the mask catalog and combined 2D spectra you can use for the project.
**Download the 2 Binospec fits files [here](https://drive.google.com/drive/folders/1HbQ6Py33Fj9rDZQ2Mb8lyFYh9iQDozHl?usp=sharing)** and put them in the `data/large_files/` directory.

## Code

An example jupyter notebook to start playing with the data.
