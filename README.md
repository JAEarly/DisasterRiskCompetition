### Open AI Caribean Challenge: Mapping Disaster Risk from Aerial Imagery ###

Competition website: https://www.drivendata.org/competitions/58/disaster-response-roof-type/page/142/

What's been tried:
- Pretrained networks (ImageNet) as feature extractors with various classifiers. Didn't work: extracted feature vectors all clumped together.
- Retraining pretrained networks (ImageNet). Using entire network. AlexNet good, ResNet152 better.

#### Running scripts ####

Execute from root directory (i.e. add root directory to PYTHONPATH).

General process:
- Use src/training/grid_search.py to train models and find hyper-parameters.
- Evaluate using src/evaluation/evaluate.py. Update models/results.md.
- If keen to submit, run src/evaluation/submit.py. 

#### Intended Project Structure ###

Based on https://drivendata.github.io/cookiecutter-data-science/#example

```
Root
|── README.md          <- The top-level README for developers using this project.
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
|
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Project documentation.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
|   └── features       <- Features extracted from models.
│
├── notebooks          <- Jupyter notebooks for exploration and communication.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting.
│
├── src                <- Source code for use in this project.
|   ├── setup.py       <- Python installation script.
│   │
│   ├── data           <- Scripts to download and extract data.
│   │
│   ├── evaluation     <- Scripts for evaluating and creating submission from models.
|   |
│   ├── features       <- Scripts to turn raw data into features for modeling.
│   │
│   ├── models         <- Scripts to train models and then use trained models to make predictions.
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations.
|
└── submissions        <- Submission files for competition upload
```
