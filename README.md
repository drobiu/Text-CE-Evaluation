# Natural Language Counterfactual Explanations in Financial Text Classification: A Comparison of Generators and Evaluation Metrics

This codebase contains the data and Python scripts for generating data, surveys, and analyzing results for the ACL GEM^2 2025 workshop submission titled Natural Language Counterfactual Explanations in Financial Text Classification: A Comparison of Generators and Evaluation Metrics.

## Codebase structure

The codebase is structured as follows:

- `software`: contains the code used for generating analyzing data in the experiments.
    - `data`: contains the raw and initally pre-processed FOMC dataset.
    - `notebooks`: contains Jupyter notebooks used for (pre-)processing the data and results of the experiments.
    - `results`: contains the pre-processed (non-aggregated) results of the experiments.
    - `raw_results`: contains the raw outputs of the counterfactual generators.
    - `survey_results`: contains the survey data processing scripts and scripts for generating the surveys. The raw survey results from the Qualtrics surveys can be downloaded from the project's [4TU.ResarchData repository](link-to-4tu) and should be placed into this folder for processing. 
    - `counterfactual_generation_scripts`: contains the scripts used to generate the counterfactuals for each generator.
        - `polyjuice`: a Jupyter notebook using the polyjuice Python package.
        - `pplm`: a Python script to be used with a cloned [PPLM](https://github.com/uber-research/PPLM) repository. 
        - `relitc`: an adapted copy of the [RELITC](https://github.com/Loreb92/relitc-counterfactuals) repository along with the training and generation scripts.<br><br>
    - `counterfactuals.csv`: generated counterfactuals.
    - `metrics_calculated.csv`: quantitative metrics calculated for each test counterfactual sentence.
    - `survey_data.csv`: factual sentences to be used in the survey.
    - `test_with_targets.csv`: factual sentences from the test set with counterfactual target classes.<br><br>
- `paper`: contains the raw `.tex` as well as the ACL style files used for our submission.