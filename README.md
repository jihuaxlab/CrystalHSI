# Rapid Identification of Defects in Organically Doped Crystalline Films via Machine Learning-Enhanced Hyperspectral Imaging

Code and data of our paper Rapid Identification of Defects in Organically Doped Crystalline Films via Machine Learning-Enhanced Hyperspectral Imaging.

## Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy==1.16.5
matplotlib==3.1.0
sklearn==0.21.3
seaborn==0.11.2
pytorch==1.7.1
xgboost==1.5.0
lightgbm==3.2.1
```

## Usage

```
python main.py
```

## Dataset

- `d1.txt`-`d5.txt` are the spectra of samples on 180, 195, 210, 225 and 240 Celsius degree, respectively, acquired by hyperspectral imaging.
- `r.txt` is the red peak's area of each spectrum.
- `g.txt` is the green peak's area of each spectrum.
- `b.txt` is the red peak's area of each spectrum.
- `l.txt` is the label of each spectrum, defined by area computation.
- `wl.txt` is the wavelength of each spectrum.