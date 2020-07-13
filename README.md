# mod-3-project-group-3-chi-sea-ds

# Table of Contents

<!--ts-->
 * [Files and Folders of Note]()
 * [General Setup Instructions](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds#general-setup-instructions)
 * [Context of Project](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds#context)
 * [Definitions](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds#definitions)
 * [Data](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds#data)
 * [Process](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds#models-used--methodology)
 * [Results](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds#results-future-investigations-and-recommendations)
 * [Next Steps](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds#future-investigations-and-recommendations)
<!--te-->

# Files and Folders of Note
```
.
├── README.md
├── data
│   ├── expanded_dataset_cm
│   ├── initial_clean_lc
│   └── raw
│       └── telecom_churn_data
├── environment.yml
├── notebooks
│   ├── exploratory
│   │   ├── 01_cm_data_exploration.ipynb
│   │   ├── 02_cm_eda_modeling.ipynb
│   │   ├── 05_cm_modeling.ipynb
│   │   ├── 06_cm_modeling.ipynb
│   │   ├── 07_cm_final_models.ipynb
│   │   ├── lmc_exploratory_nb
│   │   │   ├── 01_explore_lc.ipynb
│   │   │   ├── 02_codealong_lc.ipynb
│   │   │   ├── 03_eda_lc.ipynb
│   │   │   ├── 04_m2_lc.ipynb
│   │   │   ├── 05_state_vis_lc.ipynb
│   │   │   ├── 06_m3_lc.ipynb
│   │   │   ├── 07_clean_df_lc.ipynb
│   │   │   ├── 08_m4_lc.ipynb
│   │   │   ├── 09_notebookstats_lc.ipynb
│   │   │   ├── state_files_500
│   │   │   └── y_lc
│   │   └── model_iterations
│   │       ├── 01_cm_data_exploration.ipynb
│   │       ├── 02_cm_eda_modeling.ipynb
│   │       ├── 04_m2_lc.ipynb
│   │       ├── 05_cm_modeling.ipynb
│   │       ├── 06_cm_modeling.ipynb
│   │       ├── 06_m3_lc.ipynb
│   │       ├── 07_cm_final_models.ipynb
│   │       └── 08_m4_lc.ipynb
│   └── report
│       ├── figures
│       │   ├── churn_by_state.png
│       │   ├── cust_area_code.png
│       │   ├── cust_serv_call_churn.png
│       │   ├── dis_charge_100.png
│       │   ├── fsm_feat_import.png
│       │   ├── int_plan_churn.png
│       │   ├── tot_day_charg_churn.png
│       │   └── tot_day_charge_dist.png
│       └── final_notebook.ipynb
├── reports
│   ├── Group\ 3\ Customer\ Churn\ Presentation\ Deck.pdf
│   └── figures
└── src
    ├── cm_class_KNN.py
    ├── cm_class_LRM.py
    ├── cm_functions_balancing.py
    ├── cm_functions_preprocessing.py
    ├── cm_functions_tuning.py
    ├── data_cleaning_lc.py
    └── modelling_lc.py
```
#### Repo Navigation Links
 - [presentation.pdf](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds/blob/master/reports/Group%203%20Customer%20Churn%20Presentation%20Deck.pdf)
 - [final summary notebook](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds/blob/master/notebooks/report/final_notebook.ipynb)
 - [exploratory notebooks folder](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds/tree/master/notebooks/exploratory)
 - [src folder](https://github.com/chum46/mod-3-project-group-3-chi-sea-ds/tree/master/src)
 
# General Setup Instructions 

Ensure that you have installed [Anaconda](https://docs.anaconda.com/anaconda/install/) 

### `churn` conda Environment

This project relies on you using the [`environment.yml`](environment.yml) file to recreate the `churn` conda environment. To do so, please run the following commands *in your terminal*:
```bash
# create the housing conda environment
conda env create -f environment.yml
# activate the housing conda environment
conda activate housing
# if needed, make housing available to you as a kernel in jupyter
python -m ipykernel install --user --name churn --display-name "Python 3 (housing)"
```

# Context:

This project aims to provide SyriaTel with a model to help predict whether a customer will soon churn. In an article about churn reduction in the telecom industry by the Database Marketing Institute, it was noted that telecom companies have an annual average churn rate between 10%-67%. The article states that "industry retention surveys have shown that while price and product are important, most people leave any service because of dissatisfaction with the way they are treated". With this in mind, we aim to highlight areas where customer service could be improved. We find in out report from this dataset, that SyriaTel has a churn rate of roughly 15% in customers who have been with the company for less than 245 days.

## Aims:

This project aims to:

    - Investigate labeled data on 3333 customers who have held accounts with the company for less than 243 days.
    - Provide inferential statistics and visualisations based on this data.
    - Create predictive, supervised learning models from the data to predict churn
    
# Definitions:

    - Churn: a customer who closes their account with SyriaTel. A prediction of True relates to a customer who will churn.
    - Predictive model: A mathemaical processes which takes in data utilizes statistics to predict outcomes. 
    
# Data:

This project utilises data from the Churn in Telecom dataset from Kaggle.

The target variable in this dataset that we aimed to predict was identified as the churn column.

The features of this dataset include locational information (state and area_code) as well as plan details such as call minutes, charges, customer services calls and whether the customer had an international plan and/or voice mail plan. Our model iterations utilised subsets of these features as well as aggregations of these features to determine which features would best predict cusomter churn.

The raw, csv dataset can be downloaded directly from the kaggle website or can be found in this repo here.

# Models used + Methodology:

This project tests a variety of classification models including:

    - Decisioin Tree Classifier
    - Logistic Regression Classifier
    - KNN Classifier
    - Random Forest Classifer
    - Gradient Boost Classifer

We evaluated our models based on the recall score metric as well as the corresponding confusion matrix. Once the best model was identified, we assessed the model performance on a seperate test set to determine whether the model continued to perform well or if the model was overfitting.

The decision behind choosing to evaluate the model on recall was made by considering the cost and impact of false negative predictions, that is, we determined that it was more costly for the company for the model to predict that a customer would stay with SyriaTel when in fact that would churn/leave. This would lead to a missed opportunity for the company to dedicate retention resources towards that customer and keeping their business. Maximising recall score accounts for this scenario in our model and so it was for this reason that we chose this as our evaluation metric.

# Results, Future Investigations and Recommendations:

#### Best model:

Our best model was a gradient boost model which produced a 0.81 recall score on the test data and only 2.9% of the model's predictions on the test data were labeled as false negatives. This was a significant improvement from our FSM which had a recall score of 0.737 and 4.4% of predictions were false negatives.

The parameters of this model were:

    - loss = 'exponential'
    - learning_rate = 0.01
    - min_samples_leaf = 0.1
    - min_samples_split = 0.1
    - max_depth = 3
    - SMOTE balancing
    - StandardScaler

#### Future Investigations and Recommendations:


    - Further investigation should be devoted to looking into the other characteristics of these customers to find out why there was a need to make this many calls to customer service and how the company could better assist these customers.
    - Given that over 42% of international plan holders churn, further investigation into retention efforts for these customers might be a worthwhile effort.
    - Further investigation should be done to see what is going on in these high churn states to see what trends might be causing this.
    - Investigate ways to incentivise customers with total day charges over $55 to stay with with the company by creating added value and perks. This investigation found that 100% of these customers churn.







