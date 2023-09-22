# Membership Inference Attacks Against Synthetic Data

The code in this repository corresponds to the methodology and results as 
presented in the following two papers: 

1. ["Achilles' Heels: Vulnerable Record Identification in Synthetic Data Publishing"](https://arxiv.org/abs/2306.10308)
2. ["Synthetic is all you need: removing the auxiliary data assumption for membership inference attacks against synthetic data"](https://arxiv.org/abs/2307.01701)


## (1) Install environment

To replicate our conda environment, it should suffice to run the following sets of commands:
- Create and activate the env:
    - `conda create --name mia_synthetic python=3.9`
    - `conda activate mia_synthetic`
- Clone and install the requirements from the [reprosyn repository](https://github.com/alan-turing-institute/reprosyn):
    - `git clone https://github.com/alan-turing-institute/reprosyn`
    - `cd reprosyn`
    - `curl -sSL https://install.python-poetry.org | python3 -`
    - `poetry install -E ektelo`. To install poetry on your system we refer to [their installation instructions](https://python-poetry.org/docs/#installing-with-the-official-installer).
    - Note that, in order to get it to work for continuous attributes as well, you might look into [this raised issue](https://github.com/alan-turing-institute/reprosyn/issues/65).

Then we have to install the C-based optimized QBS:
- `cd src/optimized_qbs/`
- `python setup.py install`

## (2) Achilles' Heels: Vulnerable Record Identification in Synthetic Data Publishing

1. In order to replicate the computation of the vulnerable record identification method, 
we refer to the notebook `notebooks/Identify_vulnerable_records.ipynb`. 

2. In order to replicate the attacks for specific target records, we refer to the 
python file `Achilles_main.py` and the script `scripts/run_experiment_achilles.sh`. 

3. For all details concerning our novel target-attention attack, we refer to the code in `src/set_based_classifier`.

## (3) Synthetic is all you need: removing the auxiliary data assumption for membership inference attacks against synthetic data

In order to replicate the results of the MIAs using only synthetic data, we refer to the python file `synthetic_only_main.py` and the script `scripts/run_experiment_synthetic_only.sh`.

The code contains the functionallity to run the experiments under multiple scenarios, for details we refer to the paper.   

## (4) Setting up the data 

Throughout our contributions, we have considered two tabular datasets, referred to as UK Census and Adult. 
For ease of usability, we provide a very small sample of the original data in the `data` folder. 
We also include the corresponding metadata files that are required as input for the reprosyn pipeline to generate synthetic data. 
For full access to the respective datasets, we refer to the references in our papers. 


