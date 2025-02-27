# Additional results
Below are the full results for all four included regression models: Lasso, Ridge, Gradient Booster Trees (GBTR), and Support Vector Regressor (SVR) regression models.
![RSE](./images/RSE.png)
![SMAPE](./images/SMAPE.png)

# How to run the experiments
## Unsuitable Target Distribution
First, a directory to store the datasets in has to be defined:
```
export DATA_DIR=<path_to_datasets_dir>
```
And a directory to store the results in:
```
export RESULTS_DIR=<path_to_results_dir>
```
Then, run the experiments with:
```
python3 -m src.experiments.imbalanced_distribution <dataset>
```
where \<dataset\> is either
- the name of a dataset from src.experiments.data
- "all", which will run all datasets included in the paper

Finally, the LateX table can be generated with:
```
python3 -m src.experiments.print_results.run imbalanced_distribution -d <dataset> --latex
```

# Acknowledgements
This work received funding from the Interuniversity Special Research Fund (IBOF/21/075) and the Flemish Government under the “Onderzoeksprogramma Artifciële Intelligentie (AI) Vlaanderen” program. 
