# Dataset provenance and licences

This directory documents the publicly available datasets used in the analyses. Local copies are included only where redistribution is permitted. The repository's MIT licence applies exclusively to the original software developed for this project. Each dataset remains subject to the terms established by its original provider.

## Heart disease

- **Local files:** `heart/processed.cleveland.data`, `heart/heart-disease.names`
- **Dataset:** Heart Disease — processed Cleveland subset
- **Provider:** UCI Machine Learning Repository
- **Authors:** Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano
- **Source:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **DOI:** https://doi.org/10.24432/C52P4X
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository.

Only the processed Cleveland subset used by the replication is retained.

## Chronic kidney disease

- **Local file:** `ckd/kidney_disease.csv`
- **Dataset:** Chronic Kidney Disease
- **Provider:** UCI Machine Learning Repository
- **Authors:** L. Rubini, P. Soundarapandian, and P. Eswaran
- **Source:** https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- **DOI:** https://doi.org/10.24432/C5G020
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:** Rubini, L., Soundarapandian, P., and Eswaran, P. (2015). *Chronic Kidney Disease* [Dataset]. UCI Machine Learning Repository.

The repository contains the CSV representation used by the replication notebook.

## Pima Indians diabetes

- **Local file:** `diabetes/diabetes.csv`
- **Dataset:** Pima Indians Diabetes Database
- **Original provider:** National Institute of Diabetes and Digestive and Kidney Diseases
- **Local source:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Licence:** CC0 1.0 Universal / Public Domain
- **Associated citation:** Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., and Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Symposium on Computer Applications and Medical Care*, 261–265.

Although attribution is not required by CC0, the dataset and associated study are cited for scholarly provenance.

## HbA1c and hospital readmission

- **Analytical configuration:** HbA1c
- **Local files:** `hba1c/diabetic_data_initial.csv`, `hba1c/id_mapping.csv`
- **Original filenames:** `diabetic_data.csv`, `IDS_mapping.csv`
- **Dataset:** Diabetes 130-US Hospitals for Years 1999–2008
- **Provider:** UCI Machine Learning Repository
- **Authors:** John Clore, Krzysztof Cios, Jon DeShazo, and Beata Strack
- **Source:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **DOI:** https://doi.org/10.24432/C5230J
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:** Clore, J., Cios, K., DeShazo, J., and Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Dataset]. UCI Machine Learning Repository.

The local filenames reflect the names used in the replication workflow.

## Stroke prediction

- **Required local file:** `stroke/healthcare-dataset-stroke-data.csv`
- **Dataset:** Stroke Prediction Dataset
- **Dataset curator:** Fed Soriano
- **Source:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **Licence displayed by Kaggle:** Data files © Original Authors

The dataset file is not redistributed in this repository because its Kaggle record does not provide an explicit open-data licence. To reproduce the analysis, download `healthcare-dataset-stroke-data.csv` from the source page and place it at:

`data/stroke/healthcare-dataset-stroke-data.csv`

Access to and reuse of the dataset remain subject to the terms established by its original authors and Kaggle.

## Reuse notice

No ownership is claimed over these datasets. Dataset inclusion in this repository does not place them under the repository's MIT licence. Users must comply with the original licences, access conditions, citation requirements, and any ethical restrictions established by the respective providers.

Sources and licence information were last checked on 28 October 2026.