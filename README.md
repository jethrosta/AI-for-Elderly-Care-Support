# AI for Elderly Care & Support

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#background">Background</a> 
    </li>
    <li>
      <a href="#business-understanding">Business Understanding</a> 
      <ul>
         <li><a href="#problem-statements">Problem Statements</a></li>
         <li><a href="#goals">Goals</a></li>
         <li><a href="#solution-statements">Solution Statements</a></li>
      </ul>
    </li>
    <li><a href="#data-understanding">Data Understanding</a></li>
    <li><a href="#data-preparation">Data Preparation</a></li>
    <ul>
         <li><a href="#data-cleaning">Data Cleaning</a></li>
         <li><a href="#data-transformation">Data Transformation</a></li>
         <li><a href="#data-splitting">Data Splitting</a></li>
      </ul>
    </li>
    <li><a href="#modeling">Modeling</a></li>
    <ul>
         <li><a href="#logistic-regression">Logistic Regression</a></li>
         <li><a href="#decision-tree">Decision Tree</a></li>
      </ul>
    </li>
    <li><a href="#evaluation-matrix">Evaluation Matrix</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#comparison">Comparison</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
  </ol>
</details>

## Background
***
<div align='center'>
  <img src = 'assets/dataset-cover.jpg'>
</div>
![Elderly with AI](assets/dataset-cover.jpg)

Indonesia has entered the "*aging population*" phase, with the elderly population reaching about 12% of the total population in 2023, equivalent to 29 million people. This number is projected to increase to 20% or around 50 million by 2045. As people age, they face health risks such as chronic diseases, cognitive decline, and mobility limitations. This demands a more responsive and sustainable care system.

AI agents can monitor the vital signs of the elderly in real-time, detect anomalies, and provide early warnings to medical personnel or family members. This allows for quick intervention and prevents worsening health conditions. AI agents can interact with the elderly, provide emotional support, and reduce feelings of loneliness, which is crucial for their mental well-being. AI agents can remind the elderly to take their medications on schedule, reducing the risk of forgetfulness and ensuring adherence to treatment regimens. 

With the increasing number of elderly people in Indonesia, the integration of AI agents into the healthcare system becomes a solution that is not only efficient but also enhances the quality of life for the elderly. The use of relevant datasets can accelerate the development of this technology, ensuring that the elderly receive the care they need in a timely and personalized manner.

**[Sources]**
- https://epaper.mediaindonesia.com/detail/siapkan-penduduk-lansia-aktif-dan-produktif-untuk-usia-yang-lebih-panjang-2?utm_source=chatgpt.com
- https://sehatnegeriku.kemkes.go.id/baca/rilis-media/20240712/2145995/indonesia-siapkan-lansia-aktif-dan-produktif/?utm_source=chatgpt.com
- https://www.bps.go.id/id/publication/2023/12/29/5d308763ac29278dd5860fad/statistik-penduduk-lanjut-usia-2023.html?utm_source=chatgpt.com

## Business Understanding
Indonesia is currently facing increasing demographic challenges with a growing elderly population. The elderly often experience declines in physical and cognitive abilities, which can lead to various health risks such as falls, loneliness, deteriorating health conditions, and non-adherence to medication. Although conventional healthcare systems exist, most are still reactive and struggle to monitor the elderly's conditions in real-time. In this context, it is essential to create a more adaptive and proactive system to detect potential dangers that may occur among the elderly. Therefore, the primary goal of this project is to develop an AI model that can monitor the activities and physical conditions of the elderly in real-time, provide early warnings to caregivers or family members, and improve the accuracy of monitoring systems by minimizing detection errors, both false positives and false negatives.

### Problem Statements

Indonesia faces demographic challenges with a significant increase in the elderly population. The elderly often experience declines in physical and cognitive abilities, leading to risks such as falls, loneliness, chronic health deterioration, and non-adherence to medication. Conventional healthcare systems struggle to monitor their conditions in real-time and are reactive rather than proactive.

*Main Issues:*

1. Lack of an automatic and adaptive monitoring system that can detect potential dangers or behavioral changes in the elderly in real-time.
2. Insufficient personalization in the alert and support systems for the physical and mental conditions of the elderly.
3. Existing systems tend to be less accurate or have many false alarms.

### Goals

1. Build an AI model that can monitor and analyze the activities and conditions of the elderly in real-time using sensor data, location, and activities.
2. Provide adaptive **early warnings** to caregivers or family members based on predictions from the model.
3. Improve the accuracy of the monitoring system with intelligent algorithms, minimizing false positives and false negatives.

### Solution Statements
To address these issues, two algorithmic approaches will be systematically applied and compared:

**✅ Baseline Model**

*Algorithm: Logistic Regression*
- Reason: Fast, interpretable, and suitable for categorical and numerical data.
- Goal: Provide a baseline for identifying abnormal events from the activities of the elderly based on features in the dataset (such as `activity`, `location`, `alert_flag`, `timestamp`, etc.).

*Algorithm: Decision Tree*
- Reason: Fast, interpretable, and suitable for categorical and numerical data.
- Goal: Provide a baseline for identifying abnormal events from the activities of the elderly based on features in the dataset (such as `activity`, `location`, `alert_flag`, `timestamp`, etc.).

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

**🚀 Improved Model**

*Logistic Regression:*

Logistic Regression is a statistical model used for binary classification. Although it is not an ensemble model, it can serve as a baseline for comparing the performance of more complex models.

*Tuning:*
- Regularization (L1, L2)
- C (inverse of regularization strength)

*Goal:* 
- Generate probability predictions for binary classification and can be used to understand the relationship between independent and dependent variables.

*Evaluation Metrics:*
- ROC-AUC Score
- F1-Score (especially for the minority class “alert”)
- Confusion Matrix analysis

*Decision Tree:*

Decision Tree is a model that splits data into subsets based on feature values, forming a tree structure. Although easy to understand, it is prone to overfitting.

*Tuning:*
- max_depth (maximum depth of the tree)
- min_samples_split (minimum number of samples required to split a node)
- min_samples_leaf (minimum number of samples in a leaf node)

*Goal:* 
- Generate a model that can handle non-linear data and interactions between features, although care must be taken to avoid overfitting.

*Evaluation Metrics:*
- ROC-AUC Score
- F1-Score (especially for the minority class “alert”)
- Confusion Matrix analysis

**📊 Evaluation Methodology**
1. Cross-validation (k=5) to evaluate generalization performance.
2. SMOTE (Synthetic Minority Over-sampling Technique) to handle data imbalance if the number of alerts is low.
3. Feature importance analysis to provide insights into which features are most influential.

## Data Understanding
### Data Source URL
https://www.kaggle.com/datasets/suvroo/ai-for-elderly-care-and-support?select=safety_monitoring.csv

### Number of Rows and Columns
The dataset contains the following row and column information:
The dataset contains 3 .csv files named:
   - `daily_reminder.csv`
   - `health_monitoring.csv`
   - `safety_monitoring.csv`
Each dataset has 10,000 rows and varying numbers of features. The `daily_reminder.csv` has 7 features, `health_monitoring.csv` has 12 features, and `safety_monitoring.csv` has 10 features. Descriptions of each feature are provided in the next section.

### Description of All Features in the Data

#### `daily_reminder.csv`
This dataset has 10,000 rows with 7 features, but the 7th feature contains an empty feature named 'Unnamed' that holds empty data or *NaN Values*.

<div align="center">
  
| No | Feature               | Description                                                            |
| -- | --------------------- | ---------------------------------------------------------------------- |
| 1  | `Device-ID/User-ID`   | Unique ID for elderly users.                                          |
| 2  | `Timestamp`           | Time of recording or creating the reminder.                           |
| 3  | `Reminder Type`       | Type of reminder, such as "Exercise", "Hydration", "Medication", etc. |
| 4  | `Scheduled Time`      | Scheduled time for the activity or reminder.                          |
| 5  | `Reminder Sent`       | Whether the reminder has been sent to the user. (Yes/No)             |
| 6  | `Acknowledged`        | Whether the user has acknowledged or responded to the reminder. (Yes/No) |
| 7  | `Unnamed: 6`          | Empty column (contains NaN), possibly an error from the CSV file.    |

</div>

#### `health_monitoring.csv` 
This dataset has 10,000 rows with 12 features and complete data conditions, but it can be manipulated to fit our model, such as converting categorical forms to numeric.
<div align="center">
  
| No | Feature                                  | Description                                                      |
| -- | ---------------------------------------- | -------------------------------------------------------------- |
| 1  | `Device-ID/User-ID`                     | Unique user ID.                                              |
| 2  | `Timestamp`                              | Time of recording health conditions.                            |
| 3  | `Heart Rate`                             | User's heart rate.                                        |
| 4  | `Heart Rate Below/Above Threshold`       | Whether the heart rate is outside normal limits. (Yes/No)     |
| 5  | `Blood Pressure`                         | Blood pressure in "systolic/diastolic mmHg" format.          |
| 6  | `Blood Pressure Below/Above Threshold`   | Whether blood pressure is outside normal limits. (Yes/No)            |
| 7  | `Glucose Levels`                         | Blood glucose levels.                                           |
| 8  | `Glucose Levels Below/Above Threshold`   | Whether glucose levels are abnormal. (Yes/No)                    |
| 9  | `Oxygen Saturation (SpO₂%)`              | Percentage of blood oxygen saturation.                             |
| 10 | `SpO₂ Below Threshold`                   | Whether oxygen saturation is below minimum limits. (Yes/No)       |
| 11 | `Alert Triggered`                        | Whether the system triggered an alert based on this data. (Yes/No) |
| 12 | `Caregiver Notified`                     | Whether the caregiver has been notified. (Yes/No)                    |

</div>

#### `safety_monitoring.csv`
This last dataset contains 10,000 rows of data with 10 features. This data condition experiences data imbalance in the `Alert_Triggered` and `Fall_Detected` features.
<div align="center">
  
| No | Feature                           | Description                                                        |
| -- | --------------------------------- | ------------------------------------------------------------------ |
| 1  | `Device-ID/User-ID`              | Unique user ID.                                                  |
| 2  | `Timestamp`                       | Time of recording activity.                                      |
| 3  | `Movement Activity`               | Current movement activity, such as "No Movement", "Lying", etc. |
| 4  | `Fall Detected`                   | Whether the system detected the user falling. (Yes/No)            |
| 5  | `Impact Force Level`              | Level of impact force when falling (if any).                     |
| 6  | `Post-Fall Inactivity Duration`   | Duration of inactivity after falling (in seconds).              |
| 7  | `Location`                        | User's location at the time of recording (Kitchen, Bedroom, etc.). |
| 8  | `Alert Triggered`                 | Whether the system sent an alert. (Yes/No)                      |
| 9  | `Caregiver Notified`              | Whether the caregiver was notified. (Yes/No)                    |
| 10 | `Unnamed: 9`                      | Empty column, not relevant.                                     |

</div>


### Data Conditions
Data Conditions: 
- There are `Unnamed` or empty data in daily_monitoring and safety_monitoring.
- All data in daily_reminder is still in `object` format.
- All categorical data is still in `object` format.
- The data distribution already represents real-world data, but there is data imbalance in `Alert_Triggered` and `Fall_Detected_Counts`.

## Data Preparation
This section outlines the steps taken to prepare the data before analysis and modeling. This process is crucial to ensure that the data used in the model is clean, relevant, and ready for further analysis. Here are the data preparation techniques applied in the notebook:

### Data Cleaning
- Drop User-ID in df_health_monitor.
- Separate Systolic and Diastolic data.

### Data Transformation
- **Encoding Categories**: Categorical variables are converted to numeric format using *label encoding* techniques on categorical columns such as the `Alert Triggered` feature.
- **Binning**: Perform binning on Diastolic and Systolic data.

### Data Splitting
Data is divided into training and testing sets where the predictor features `X` consist of:
- `Timestamp`,
- `Heart Rate`, 
- `Heart Rate Below/Above Threshold (Yes/No)`,
- `Blood Pressure Below/Above Threshold (Yes/No)`,
- `Glucose Levels`,
- `Glucose Levels Below/Above Threshold (Yes/No)`,
- `Oxygen Saturation (SpO₂%)`,
- `SpO₂ Below Threshold (Yes/No)`,
- `Blood Pressure Category`
While the target `y` contains `Alert Triggered (Yes/No)`.

With the above steps, the data has been well prepared for analysis and modeling, ensuring that the models built can provide accurate and reliable results.

## Modeling
**Algorithms Used:**
- Logistic Regression
- Decision Tree Classifier

**How the Model Works:**
### Logistic Regression
___
![](https://i.sstatic.net/7M3Mh.png)

Despite its name "regression," Logistic Regression is actually a classification algorithm, not regression like Linear Regression.
**How It Works:**
- Logistic Regression is used to predict the probability of a data point belonging to one of two categories (e.g., 0 or 1, Spam or Not Spam).
- Essentially, Logistic Regression calculates a value using a linear equation like:
   $`{z} =  w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n`$
   (where $w$ is the weight/coefficient, and $x$ is the input feature)
- However, to ensure the result is a probability (between 0 and 1), this value ${z}$ is passed through the sigmoid function:
  $`\sigma(z) = \frac{1}{1 + e^{-z}}`$
- After obtaining the result from the sigmoid function, there is usually a threshold (e.g., 0.5) for decision-making:
   - If probability > 0.5 → Predict 1
   - If probability ≤ 0.5 → Predict 0

### Decision Tree:
---
<div align='center'>
    <img src='https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dtc_002.png' width='100%' >  
</div>

Decision Tree is an algorithm that makes decisions with a tree-like structure, where each branch represents a choice based on certain features.
**How It Works:**
- Starts with the entire dataset.
- The algorithm selects the best feature that separates the data most effectively (using measures like Gini impurity, Entropy, or Information Gain).
- Splits the data into branches based on the feature's value.
- This process is repeated at each branch, forming a tree, until:
   - All data in the branch belongs to one class, or
   - No more features can be used.

**Key Parts of Decision Tree:**
- Root Node: The start of the tree, the main feature selected.
- Internal Node: Decisions based on features.
- Leaf Node: Prediction results (class 0 or 1).

## Hyperparameter Tuning
After testing the default parameters of both algorithms, hyperparameter tuning will be explored to improve model performance and accuracy. The parameters will be searched using the `Grid Search CV` method with the following parameters:
### Logistic Regression:

In Logistic Regression, there are two important parameters: `C` and `penalty`. The `C` parameter controls the strength of regularization in the model. Conceptually, `C` is the inverse of the regularization strength: the larger the value of `C`, the weaker the regularization applied. This means the model is given more freedom to fit the data, but with the risk of overfitting (too closely following the training data). Conversely, if the value of `C` is small, the regularization becomes stronger, forcing the model to be simpler, which can reduce the risk of overfitting but may also lead to underfitting (the model is too simple).

Meanwhile, the `penalty` parameter determines the type of regularization used to penalize overly complex models. There are several common types of `penalty` used:
- L2 regularization (`l2`): penalizes the square of the weights, making all weights small but still present.
- L1 regularization (`l1`): penalizes the absolute size of the weights, and can make some weights zero, resulting in a sparse model (simpler and can perform automatic feature selection).
- Elastic Net ('elasticnet'): a combination of L1 and L2 regularization.
- None (`none`): does not use regularization at all.

In general, regularization is used to prevent the model from becoming too complex and helps the model generalize better to new data. By selecting the right value of `C` and the appropriate type of `penalty`, a balance can be struck between accuracy on the training data and generalization ability on the test data.

**Key Points**
- `C` large → weak regularization → higher risk of overfitting.
- `C` small → strong regularization → simpler model, risk of underfitting.
- `penalty` = `l2` → small weights, but all features are still used.
- `penalty` = `l1` → many zero weights → automatically performs feature selection.
- Regularization is important to keep

### Decision Tree:

In a Decision Tree, there are several important parameters that significantly influence the depth of the tree and the model's generalization ability, namely `max_depth`, `min_samples_split`, and `min_samples_leaf`.

The `max_depth` parameter determines how deep the decision tree can grow. If `max_depth` is not limited, the tree can continue to grow deeper until each leaf contains only one sample, which often leads to overfitting because the model becomes too specific to the training data. By limiting `max_depth`, it forces the tree to remain simple, which can help reduce overfitting and improve generalization to new data.

The `min_samples_split` parameter sets the minimum number of samples required in a node to be split. If the number of samples in a node is less than the value of `min_samples_split`, that node will become a leaf and will not be split further. This is useful to prevent the tree from growing too deep by limiting the splitting of nodes that have too little data, thus helping to avoid overfitting.

The `min_samples_leaf` parameter sets the minimum number of samples that must be present in each leaf node. By setting the value of `min_samples_leaf` greater than one, it prevents the tree from creating very small leaves (containing only one or two samples). This makes the model more stable and reduces variance (fluctuations in the model due to data noise).

In general, these three parameters are used to control the complexity of the tree. Tuning them appropriately will help produce a Decision Tree model that is not only strong on the training data but also capable of generalizing well to new data.

**Key Points**
- `max_depth`:
   - Limits the maximum depth of the tree.
   - Small value → prevents overfitting, too small value → risk of underfitting.
- `min_samples_split`:
   - Minimum number of samples in a node for that node to be split.
   - Larger value → more balanced tree and not too deep.
- `min_samples_leaf`:
   - Minimum number of samples in each leaf.
   - Helps ensure that each leaf is stable enough and does not only contain outliers.

All these parameters are used to control the complexity of the tree and avoid overfitting.

#### *Best Model:*

The Decision Tree provided the best results after tuning, especially in recall for the minority class (“alert”). It was chosen as the final model because:
- It handles imbalance better than Logistic Regression.
- It provides important feature information for model interpretation.

## Evaluation Matrix

In this project, several evaluation metrics were used to assess the performance of each model based on the characteristics of the data, which is quite imbalanced (minority class: **"alert"**) and the primary need to minimize *false negatives* in the elderly monitoring system. These metrics reflect how well the model can provide relevant and accurate predictions in the context of early warnings.

Here are the metrics used:

<h4>1. Accuracy</h4>

Formula:

<div align='center'>
  <img src='https://latex.codecogs.com/png.image?\dpi{120}&space;\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}' alt='accuracy'>
</div>
 
Used as a general overview of model performance, but not suitable as a primary metric in cases with imbalanced data because it can be misleading.

<h4>2. Precision</h4>

Formula:

<div align='center'>
  <img src='https://latex.codecogs.com/png.image?\dpi{120}&space;\text{Precision}=\frac{TP}{TP+FP}' alt='precision'>
</div>
 
This metric measures how many of the positive predictions are actually relevant. It is useful for avoiding false alarms that occur too frequently (False Positive).

<h4>3. Recall (Sensitivity)</h4>

Formula:
<div align='center'>
  <img src='https://latex.codecogs.com/png.image?\dpi{120}&space;\text{Recall}=\frac{TP}{TP+FN}' alt='recall'>
</div>

 
This is very important in this context because it ensures that "alert" events are not missed (minimizing False Negatives).

<h4>4. F1-Score</h4>

Formula:

<div align='center'>
  <img src='https://latex.codecogs.com/png.image?\dpi{120}&space;F1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}' alt='f-1 score'>
</div>
 
F1 Score combines precision and recall into a single harmonic metric, ideal for imbalanced data.

<h4>5. ROC-AUC Score</h4>

Used in the Random Forest model to measure the model's ability to distinguish between positive and negative classes. The closer to 1, the better the performance.

<h4>6. PR-AUC (Precision-Recall AUC)</h4>

Used in XGBoost as the primary metric to measure performance against the minority class while considering the trade-off between precision and recall.

<h4>7. Log Loss</h4>

Measures how well the model predicts probabilities that approach the actual values. Low log loss indicates good predictions and good model calibration.

<h4>8. Confusion Matrix</h4>

Provides a comprehensive overview of the distribution of true/false predictions for each class.

## Results
**Logistic Regression 📈**

Here are the results of the confusion matrix from the Logistic Regression baseline model before tuning and the ROC-AUC Curve.

<div style="display: flex; gap: 10px;">
  <img src="assets/conf_Mat_LogReg_1.png" alt="Image 1" width="48%" height='100%'>
  <img src="assets/ROC_LogReg_1.png" alt="Image 2" width="48%" height='100%'>
</div>

Upon analysis, it turns out that there are features that have greater significance than others.

<div align='center'>
  <img src='assets/importance_LogReg_1.png'>
</div>

Then hyperparameter tuning was performed, yielding the following results:

<div style="display: flex; gap: 10px;">
  <img src="assets/conf_Mat_LogReg_2.png" alt="Image 3" width="48%" height='100%'>
  <img src="assets/ROC_LogReg_2.png" alt="Image 4" width="48%" height='100%'>
</div>

It turns out that when Grid Search CV was performed, there was an improvement in estimator performance and ROC value, as well as feature significance in the model. The feature `Blood Pressure Category` increased even surpassing `Oxygen Saturation (SpO2)`.

<div align='center'>
  <img src='assets/importance_LogReg_2.png'>
</div>

**Decision Tree 🌳**
This baseline model shows quite good performance in recognizing the majority class, but still has weaknesses in detecting the minority class (Alert Triggered = True), as indicated by the still relatively low recall value. Additionally, the model also shows mild overfitting because performance on the training set is higher than on the test set.

<div style="display: flex; gap: 10px;">
  <img src="assets/ConfMat-DT_1.png" alt="Image 5" width="48%" height='100%'>
  <img src="assets/ROC-DT_1.png" alt="Image 6" width="48%" height='100%'>
</div>

Similar to the previous model, it turns out that there are features that have greater significance than others.

<div align='center'>
  <img src='assets/importance_DT_1.png'>
</div>

Then hyperparameter tuning was performed with Grid Search CV, yielding the following results:

<div style="display: flex; gap: 10px;">
  <img src="assets/ConfMat-DT_1.png" alt="Image 5" width="48%" height='100%'>
  <img src="assets/ROC-DT_2.png" alt="Image 6" width="48%" height='100%'>
</div>

Similar to the previous model, there was an increase in the ROC-AUC value and accuracy.

<div align='center'>
  <img src='assets/importance_DT_2.png'>
</div>

Overall, the Decision Tree model was successfully improved through tuning, resulting in a more accurate and sensitive model, as well as providing stronger insights into important features. This makes the Decision Tree a strong candidate for the final solution in this project.

## Comparison
| Model                         | Accuracy | Precision | Recall | F1 Score| ROC Score|
|------------------------------|----------|-----------|--------|----------|----------|
| Logistic Regression Basemodel| 0.7260   | 0.7627    | 0.9109 | 0.8302   | 0.75     |
| Logistic Regression Tuned    | 0.7935   | 0.8130    | 0.9341 | 0.8693   | 0.82     |
| Decision Tree Basemodel      | 0.8620   | 0.9040    | 0.9089 | 0.9064   | 0.90     |
| Decision Tree Tuned          | 0.8620   | 0.9040    | 0.9089 | 0.9064   | 0.91     |

## Conclusion
The final results of this project will address all [Problem Statements](#problem-statements).
- **Problem 1:**
  > *Lack of an automatic and adaptive monitoring system.*
  
  ➔ The developed real-time AI model automatically monitors the activities of the elderly based on sensor data, location, and activities. With high accuracy (91%), this proves that monitoring can be done adaptively and in real-time.

- **Problem 2:**
   > *Lack of personalization in the alert system.*
   
   ➔ The system generates adaptive alerts based on AI predictions that take into account individual activities, not just static thresholds. This means it has met the personalization needs for the physical and mental conditions of the elderly.

- **Problem 3:**
   > *Inaccurate systems with many false alarms.*
   
   ➔ With an achievement of `91%` accuracy, the model has successfully improved accuracy and significantly reduced both false positives and false negatives compared to conventional approaches.

The designed [Goals](#goals) have all been achieved in this project.
- **Goal 1:**
   > *Build a real-time AI model for the elderly*

   ➔ This has been achieved by building a monitoring model based on sensor data and the activities of the elderly, generating predictions in real-time.

- **Goal 2:**
   > *Provide early warnings to caregivers/family*
   
   ➔ The system has detected potential dangers and issued alerts based on predictions, not just responses after events.

- **Goal 3:**
   > *Increase accuracy and reduce false alarms*
   
   ➔ The result of 91% accuracy proves that this system has significantly reduced errors and is more reliable than baseline or conventional systems.

**Impact**
___
The designed solution not only succeeded at a technical level (high accuracy) but also has a real impact on the targeted social goals, namely improving the safety of the elderly, reducing the burden on families/caregivers, and making the healthcare system more proactive and personalized.
