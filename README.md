# Project 4: Shark Tank Machine Learning Model
## Authors: 
Andrea Ceriati, Kathryn Lesh, Emily Sims, and Scott Zuckerbrow. 

# Overview
Our team analyzed data from seasons 1 - 15 of Shark Tank. This is a popular TV show in the U.S. where entrepreneurs pitch their innovative ideas and products to a group of investors (referred to as 'sharks'). The goal of the pitch is to secure the funding needed to create a successful startup. Our project goal was to create a machine learning model that could predict whether a pitcher would secure a deal. 

# Data Analysis 
## About the Data
Our team sourced the Shark Tank dataset from <a href=https://www.kaggle.com/datasets/thirumani/shark-tank-us-dataset>Kaggle</a>. 
Before creating and running the machine learning model, we did exploratory data analysis in Python and Tableau to see if there were correlations or outliers that might impact modeling later in our project. 

Based on the data, 61% of pitches succeeded in securing a deal. 
One interesting outlier was the success rate of securing a deal for the Automotive Industry. While there were only 17 pitches for products categorized as Automotive, 76% of those were successful. 

Top 5 industry success rates:

<img height=300px src="img/success rate per industry.png">

As demonstrated in the graph above, the automotive industry success rate was significantly higher than the other industry categories. 



Industry success by season:

<img height=300px src="img/industry over seasons.png">

This graphic illustrates the success of each industry through various seasons. We could see some qualitative insights. For example, Season 12 (aired October 2020 during the COVID-19 pandemic) had a large uptick in deals in the Food and Beverage industry, and increase in lifestyle/home, and a decrease in health/wellness. 

## Cleaning process: 
To clean the data, we looked for columns with a significant number of missing values. The columns "City", "Multiple Entrepreneurs", "Loan", "Pitchers State", and "Pitchers Average Age" columns had a significant number of values missing, so they were dropped from the data frame. The "company website" column had almost 600 values missing, so we turned that column into a binary feature, where 0 represents 'no company website' and 1 represents ‘has company website'. 

For columns with only a few values missing, we conducted additional research to help fill in the gaps. There were 7 missing values in the "pitcher's gender" column, so we did some quick googling of the pitcher's identity and used .loc to replace the missing values with the actual values found in our google search. We updated simple formatting issues such as converting the "Season Start", "Season End", and "Original Air Date" columns to datetime format. Finally, we created dummy variables for categorical columns included in the features. 

Clean data frame sample: 

<img height=400px, src="img/dataframe sample.png">

This data frame has 1345 rows and 51 columns. 

# Model development

## Building the Model:
Before diving into model selection and development, we performed a thorough analysis of all features within the dataset. We performed a comprehensive exploratory data analysis using visuals, statistical tests such as chi-square, and ran different models, that led us to the following features as most likely to be explanatory: 'Original Ask Amount', 'Original Offered Equity', 'Has_Website', 'Pitchers Gender_Female', 'Pitchers Gender_Male', and 'Pitchers Gender_Mixed Team'.

Our target for the model was whether the pitch would get a deal on Shark Tank based on the selected features. (Target variable: Got Deal) 

## Choice of Models:
 - Logistic Regression: We used this model for its simplicity and interpretability. It is well suited for binary classification tasks like prediction of whether or not a pitch got a deal. 
 - K-Nearest Neighbors (KNN): We tried KNN because it is a non-parametric method that makes minimal assumptions about the data distribution. It works well when the decision boundary is non-linear or complex. By considering the similarity of instances, KNN can classify pitches based on their proximity to other successful or unsuccessful pitches.
 - Deep Neural Network (DNN): DNN is a powerful choice capable of learning complex patterns and relationships in the data. Its multiple layers of neurons allow it to capture hierarchical features. This is particularly useful when dealing with high-dimensional data like the features extracted from Shark Tank pitches. The DNN's flexibility makes it suitable for capturing both linear and non-linear relationships in the data.
 
## Feature selection and optimization:
To optimize model performance, we used automated sequential feature selection with the goal of systematically choosing a subset of features to optimize model performance while mitigating overfitting. The methodology involves iteratively adding or removing features based on their impact on the model's performance. One starts with an empty set of features, and then features are gradually added, with only the most informative ones retained. The scoring metric is f1, taking into account both precision and recall. This technique offers several advantages, including the reduction of feature space dimensionality, which can lead to improved model performance and decreased computational complexity.

To find the best hyperparameters for our DNN model we employed hyperparameter tuning using Keras Tuner. This approach systematically explores the hyperparameter space to identify the configuration that maximizes model performance. Key hyperparameters, including activation functions, layer count, and neuron count per layer, were optimized for the DNN model. These selections significantly influence the model's capability to discern intricate patterns within the data. Tuning hyperparameters enhances the model's efficacy in generalizing to unseen data, thereby bolstering its robustness in real-world applications.

# Findings:
## Logistic Regression:

For the logistic regression, we used sequential feature selector. The sequential feature selector forward selection starts with an empty set of features and adds the most relevant feature at each iteration until a stopping criterion is met (in this case reaching a specified number of features (5)). 

### Results: For all 5 Features
Precision: When the model predicts a class, it is correct about 59% of the time for class 0 (no deal) and 62% of the time for class 1 (deals).

Recall: The model captures only 8% of the instances of class 0 (no deal) but captures 97% of the instances of class 1 (deals).

F1-score: For class 0 (no deal), it's 13%, and for class 1 (deals), it's 75%. The F1-score balances precision and recall. It is a better measure than accuracy when dealing with imbalanced classes.

Support: The number of actual occurrences of each class. There are 133 instances of "Did Not Get Deal" and 204 instances of "Got Deal."

Accuracy: Overall, the model correctly predicts 61% of the instances.

The confusion matrix shows the counts of true positives, false positives, true negatives, and false negatives. Our model correctly predicted 10 instances of "Did Not Get Deal" and incorrectly predicted 123 instances of "Did Not Get Deal" as "Got Deal".


Overall, we found this model was unsuccessful due to the the inability to accurately predict negatives. The model performs significantly better at predicting "Got Deal" instances, as indicated by higher precision, recall, and F1-score for class 1. However, there is room for improvement, especially in capturing instances of "Did Not Get Deal" (class 0), as evidenced by the low recall and F1-score for that class.


Classification Report for All 5 Features:

<img height=300px src="img/logistic regression class report.png"> 

## KNN

### Results: For 4 Features
Precision is the proportion of true positive predictions among all positive predictions. For class 0 (no deal), it's 52%, and for class 1 (deals), it's 64%. Higher precision indicates fewer false positives. When the model predicts a company will get a deal on Shark Tank, it is correct 64% of the time.

Recall is the proportion of true positive predictions among all actual positive instances. For class 0 (no deal), it's 25%, and for class 1 (deals), it's 85%. Higher recall means the model had fewer false negatives.

F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall. For class 0 (no deal), it's 34%, and for class 1 (deals), it's 73%.

Support: The number of actual occurrences of the class in the specified dataset. For class 0 (no deal), it's 133, and for class 1 (deals), it's 204.

Accuracy is the proportion of correct predictions among the total number of predictions. It indicates the overall performance of the model. At 61%, it is sufficiently accurate. 

The macro average is the unweighted mean of precision, recall, and F1-score across classes, while the weighted average considers the support for each class. 

Overall, this model performs better for class 1 (deals) than for class 0 (no deal), as indicated by higher precision, recall, and F1-score for class 1. However, the overall performance, especially in terms of accuracy, could be improved.

Classification Report:

<img height=300px src="img/KNN classification report.png">


## Neural Network:
### Model 1
First, we designed a simple Neural Network to evaluate its effectiveness at making predictions. 

#### NN Model 1 composition: 
  First hidden layer:

  nn.add(tf.keras.layers.Dense(units = 20, activation='tanh', input_dim=number_input_features))
  
  Output layer:
  
  nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


#### NN Model 1 Summary:

<img height=300px src="img/NN model 1 summary.png">


#### NN Model 1 Results: 
Loss: 0.66

Accuracy: 0.62

### Model 2
Then to get the best hyperparameters, we ran the Keras Tuner search. 

Keras Tuner Search Results:

<img height=300px src="img/Keras Tuner search results.png">

Based on the results, we compiled a new model to run test data and compare with the initial model findings. 

#### NN Model 2 Composition: 
First hidden layer:

  nn_best.add(tf.keras.layers.Dense(units=16, activation='tanh', input_dim=number_input_features))
  
  Additional hidden layers:

  nn_best.add(tf.keras.layers.Dense(units=1, activation='tanh'))
  nn_best.add(tf.keras.layers.Dense(units=16, activation='tanh'))
  nn_best.add(tf.keras.layers.Dense(units=16, activation='tanh'))
  nn_best.add(tf.keras.layers.Dense(units=11, activation='tanh'))
  nn_best.add(tf.keras.layers.Dense(units=1, activation='tanh'))
  
Output layer:
  nn_best.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#### NN Model 2 Summary: 


<img height=350px src="img/NN model 2 summary.png">


#### NN Model 2 Tuned results:
Loss: 0.66

Accuracy: 0.61


### NN Results: Overall
After comparing the two NN models, the simple first model had a slightly higher accuracy rate. However, the NN Model 2 Tuned had a slightly lower loss rate. 

# Conclusion:
Overall, the Neural Network Keras Tuned Model 2 is the most successful as it has the lowest loss and a similar accuracy rate to the other models. 

While none of the Machine Learning models we created had an accuracy rate of over 75%, we think this is to be expected, as there are several human and psychological factors that may impact a shark’s decision on whether to offer a deal. For example, the pitcher’s charisma or ‘professional appearance’ could be influencers. 

# Sources:
Thirumani. (2024). Shark Tank US Dataset. Retrieved from Kaggle: https://www.kaggle.com/datasets/thirumani/shark-tank-us-dataset
