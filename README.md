# imzml2np

### Extract ion images from imzml and save to numpy array


### Setup
```
git clone https://github.com/Buglakova/imzml2np
cd imzml2np
python -m venv imzml2np
source imzml2np/bin/activate
pip install -r requirements.txt
pip install -e .
```
### Usage
```
python extract_ion_images.py sample.imzML metadata.csv ion_images.npy --tol 5 --base_mz 200
```

`metadata.csv` must have `mz` column, otherwise it's possible to use function `load_db` to calculate `mz` given formula, adduct and charge. There may also be other columns. 