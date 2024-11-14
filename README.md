# create and activate a fresh environment named ps3

# see environment.yml for details

mamba env create
conda activate ps3

pre-commit install
pip install --no-build-isolation -e .

```

```
