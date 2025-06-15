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
