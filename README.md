# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
The process is done using two methods: 
1- Optimizing the hyperparameters of a standard Scikit-learn Logistic Regression using HyperDrive 
2- Build and optimize the model with AutoML. 
Both the models are built on the same dataset so that they can be compared afterwards.

## Summary
This dataset contains data about bank marketting we seek to predict if the client will subscribe to a term deposit with the bank. (y= Yes or No)

The best performing model is the model with the higher acuuracy.

## Scikit-learn Pipeline
**Our pipeline architecture, includes reading data, hyperparameter tuning, and classification.** The data is first read as a tabular dataset, then it is split to a training and a testing dataset, we chose random sampling for parameter tuning and logistic regression as a classification algorithm.

**Random sampling** supports early termination of low-performance runs. Also supports both discrete and continuous hyperparameters. In random sampling, hyperparameter values are randomly selected from the defined search space with two hyperparameters '--C' (Reqularization Strength) and '--max_iter' (Maximum iterations to converge).
 In this experiment the defined spaces are, -C (inversion of regularization strength): uniform (0.01, 1), ie, It returns values uniformly distributed between 0.01 and 1.00. -max-iter (maximum number of iterations): choice (100, 150, 200, 250, 300), ie, It returns a value chosen among given discrete values 100, 150, 200, 250, 300.

**Early termination policy**  Bandit is an early termination policy that terminates any runs where the primary metric is not within the specified slack factor with respect to the best performing training run. In our case BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5). Slack factor refers to the slack allowed with respect to the best performing training run in ratio, ie, if the best metric run is less than 0.909, it will cancel the run. evaluation_interval specifies the frequency for applying the policy, ie, everytime the training script logs the primary metric count, policy is applied. delay_evaluation specifies the number of intervals to delay the policy evaluation. The policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation. In our case after 5 intervals the policy is delayed.

## AutoML
The best performing model was a model with an accuracy of aproximately .91627 using VotingEnsemble algoritim and that took 0:01:24 to run.

## Pipeline comparison
The best performing model with Hyperdrive Pipeline was a model with acuuracy aproximately : 0.91578 and with Automl Pipeline, the best performing model was a model with acuuracy approximately :0.91627. However, AutoML avoids the need of frequrent changes in script for diffrent algoritihms. Check the images folder for more details.

HyperDrive Pipeline: Classification technique used: Logistic Regression
Best Run Selection: Selected out of multiple runs (same algorititm with different hyperparameters)
Data Pre-Prossesing: The data is cleaned with the clean_data() imported from train.py, rows with missing values are dropped and categorical(textual) fields converted to numerical fields. 
Accuracy: .91578.

AutoML Pipeline: Classification technique used: Multiple Alogritims 
Best Run Selection: Selected out of multiple runs with diffrent algorititms with there auto generated hyperparameters. 
Data Pre-Prossesing: The data is cleaned with the clean_data() function imported from train.py, with rows with missing values dropped and categorical(textual) fields converted to numerical fields. 
Accuracy: .91627

## Future work
- In the case of scikit-learn based model, a different parameter sampler (Grid sampling or Bayesian sampling) can be used. Early stopping is mainly for iterative solutions, like Grid sampling. It is mainly used to terminate processes when a run gets stuck and does not improve over a couple of iterations. Analyze the performance by removing the bandit policy. Other parameters (max_total_runs, max_concurrent_runs, primary_metric) can be changed to optimize the pipeline.
- In the case of AutoML run, we got an imbalanced data warning. This can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class. Run the AutoML process after recifying the same. Also try to run the AutoML run longer by removing the 30mins time limit.

## Proof of cluster clean up
1- Compute cluster deleted via code (Check images).
2- Virtual Machine delteted Manually.
