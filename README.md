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

- `d.txt` is the spectra of samples, acquired by hyperspectral imaging.
- `r.txt` is the red peak's area of the each spectrum.
- `g.txt` is the green peak's area of the each spectrum.
- `b.txt` is the red peak's area of the each spectrum.
- `l.txt` is the label of the each spectrum, defined by area computation.