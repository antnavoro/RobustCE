# RobustCE
Code to generate counterfactual explanations based on the paper "Robust counterfactual explanations in classification and regression" by Emilio Carrizosa, Antonio Navas-Orozco. The paper can be found here: https://www.sciencedirect.com/science/article/pii/S0377221726001888

For the results of the experiments contained in this repository, Python 3.12.8 and Gurobi 12.0.1 were used. These experiments were conducted on a MacBook Pro equipped with an Apple M4 Pro chip (12 cores: 8 performance and 4 efficiency), 48 GB of RAM, and running macOS Sequoia 15.1 (64-bit).

Description of the files:
- File robustCE.py contains the code to compute robust counterfactual explanations. The code starts with the selection of GLM (and with it, its associated dataset). After loading the dataset, the adjustable parameters are defined. Later, several necessary functions are defined. At the end, the code is set to parallelly compute robust CEs to different individual in a one-for-one fashion. Setting kappa to 1 or 0, and modelType to logistic, probit, linear or Poisson, allows for the reproduction of the experiments from the paper. The results are set to be saved in a csv format. Files Poisson_kappa_0.csv, Poisson_kappa_1.csv, linear_kappa_0.csv, linear_kappa_1.csv, logistic_kappa_0.csv, logistic_kappa_1.csv, probit_kappa_0.csv and probit_kappa_1.csv are the results of running robustCE.py.
- File plot_data.py contains the code to generate the figures that are included in the paper. These can be consulted in the folder paper_fullSize in png format, and in folder figures_vector in pdf format.
- File load_dataset.py contains the code to load the data sets, which is needed in both robustCE.py and plot_data.py. The raw data it loads is contained in files SeoulBikeData.csv, breast-cancer-wisconsin.data, communities.data and communities.names.

To run the experiments on the same conditions:
# check python version
python --version

# install requirements
pip install -r REQUIREMENTS.txt

# install Gurobi
from https://www.gurobi.com/products/gurobi-optimizer/
