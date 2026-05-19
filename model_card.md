# Model Card

## Model Details
Emily Franzer created the model. It is a logistic regression using the default hyperparameters in scikit-learn 1.5.1.

## Intended Use
This model should be used to predict whether or not an individual's salary is greater $50,000. 

## Training Data
The data was obtained from the UCI Machine Learning Repository.
The original data set has 48842 rows. The dataset provided for use for this project included 32561 rows. An 80-20 split was used to break this into a train and test set; the training set consisted of 26048 rows. 
No stratification was applied. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels. Continuous data was scaled with a standard scaler. 

## Evaluation Data
The test set was made up of 6513 rows (20% of the data). The same preprocessing used in training was applied. The target label for this data was salary, with a goal of identifying individuals with salaries above $50,000. 

## Metrics
The model was evaluated using precision, recall, and F1 score. The precision is 0.7329. The recall is 0.5900. The F1 score is 0.6537. 
These metrics were also measured across slices of categorical features, the results of which are contained in slice_output.txt. 

## Ethical Considerations
Use of this data for real-world present-day applications would not be representative, as the data is from the 1994 Census and is therefore outdated. 
In addition, it is known that the dataset is skewed toward white males, meaning that the model likely would not reliably generalize to underrepresented groups. 

## Caveats and Recommendations
For real-world, present-day applications, users should seek out a more current dataset, as this data is over 30 years old. For more thorough analysis and understanding of the present model, slice metrics should be reviewed, as performance differed across demographic slices. 