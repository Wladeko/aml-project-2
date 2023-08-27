# Advanced Machine Learning Project 2
Repository for Advanced Machine Learning course second project. 

My main AML course repository can be found under this [link](https://github.com/Wladeko/advanced-machine-learning).

---
## Description
The aim of the project is to compare different feature selection methods. The goal is to propose methods of feature selection and classification, which allow to build a model with large predictive power using small number of features.

Full project task is located in `resources` subdirectory.

---
## Project structure
```
.
├── README.md
├── .gitignore
├── .pre-commit-config.yaml <- pre-commit hooks for standardizing code formatting
├── .project-root <- root folder indicator
│
├── notebooks
│   └── ...
│ 
├── requirements.txt <- python dependencies
│ 
└── src
    ├── data <- data loading and preprocessing
    │   ├── artificial.py <- artificial dataset
    │   └── spam.py <- spam dataset
    │   
    └── evaluate.py <- implementation of evaluation methodology

```
---
## Results
We presented obtained results in short [report](https://github.com/Wladeko/aml-project-1/blob/main/report.pdf).

In the final tally, during inference on the test set, our proposed solution turned out to be the best among all students in our year. We achieved:

- Score `0.9` for artificial test dataset
- Score `0.96` for spam test dataset

---
## Co-author
Be sure to check co-author of this project, [Lukas](https://github.com/ashleve).