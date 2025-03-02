Project Title: **An Integrated Machine Learning Approach to Understanding Fishing Fleet Dynamics in Croatia: Classification and Clustering for Economic Insights (2011-2022)**


**Project Description**
The project was initiated with an aim to analyze fishing fleet dynamics in Croatia through data cleaning, clustering, classification, and regression techniques. The predictions to be made here are namely Total Fishing Fleet Vessel Count classification into Low, Medium, and High, and the investigation for any hidden data patterns through unsupervised clustering analysis. The datasets were sourced from Eurostat, and they include economic indicators related to fisheries and aquaculture in Croatia.

**Techniques Used**

**1. Data Cleaning and Preprocessing**

The project integrated data from eight different datasets concerned with various aspects of Croatian fisheries, including:

Catch Weight (TLW)
Input Weight for Capture-based Aquaculture (TLW)
Aquaculture Production (TLW)
Aquaculture Production at Juvenile Stage (MIO)
Landings of Fishery Production (TPW)
Fishing Fleet in unit (GT)
Total Fishing Fleet Power (KW)
Total Fishing Fleet Vessel Count (NR)

**Major Preprocessing Steps **

Merging the datasets on TIME_PERIOD key.
Dropping irrelevant columns against redundant data.
Rounding up numerical values to 2 places after the decimal.
Deleting duplicates and dealing with missing values (if present). 
Outlier Detection and Treatment: Outliers were detected by the Interquartile Range (IQR) and capped.
Regression Analysis (Prediction Task)
**Aim**: Prediction of Total Fishing Fleet Vessel Count (NR) using selected numerical features.

**Selected Features:**

Fishing Fleet in unit (GT)
Catch Weight (TLW) 
Aquaculture Production at Juvenile Stage (MIO)
Input Weight for Capture-based Aquaculture (TLW)
Landings of Fishery Production (TPW)
Total Fishing Fleet Power (KW)

The following algorithms were tried:

Linear Regression,
Random Forest Regressor,
Gradient Boosting Regressor.


**Evaluation Metrics:**

Mean Squared Error (MSE), 
R^2 Score.

**The greatest model:**

Gradient Boosting Regressor where R² = 0.9174.

**3. Classification Analysis (Categorization Task)**
**Goal**: Classify total fishing fleet vessel counts (NR) into:

Low
Medium
High

**Methodology:**

Percentile-based binning (33% quantiles) was applied to form 3 fleet categories.
Top features selected for classification based on correlation analysis:
Fishing fleet in unit (GT)
Landings of fishery productions (TPW)
Total fishing fleet power (KW)
**Algorithms applied:**

Support vector machine (SVM) - One-vs-Rest-multi-class strategy
Naive bayes
Logistic regression
K-Nearest Neighbors (KNN)

**Evaluation Metrics:**

Accuracy
F1 Score
Precision
Recall
Confusion Matrix
ROC Curve-multi-class
**Best Model:**
✅ **Logistic Regression** was the most balanced between accuracy, precision, recall, and F1 score.

**4. Clustering Analysis (Unsupervised Pattern Discovery)**
 The objective is to explore for natural groupings or hidden patterns in the fleet data so as to increase the understanding concerning economic behavior. 

**Approach:**

The three most correlated features selected for clustering:

Total Fishing Fleet Power (KW)
Fishery Production Landings (TPW)
Fishing Fleet in unit (GT)

Principal Component Analysis (PCA) was performed for dimensionality reduction before clustering.

K-means Clustering was run to cluster the data.

**Cluster Validation Techniques:**

The Elbow Method: To find the optimal number of clusters.  
Silhouette Score: To provide evidence for clustering quality.  

**Key Finding:**

✅The number of optimum clusters was 3, which fits the real-world intuition about the different fleet size groups (small, medium, large).

**Tools & Libraries Used**

Python 3 (Core language)
Pandas (Data Manipulation)
NumPy (Computing)
Matplotlib and Seaborn (Visualization)
Scikit-learn (ML + Clustering)
GridSearchCV (Hyperparameter Tuning)
SciPy (Statistical Analysis)
PCA (from sklearn.decomposition) (Dimensionality Reduction)

**Folder Structure**
/project-folder/
|-- datasets/
|   |-- dataset_one.csv
|   |-- dataset_two.csv
|   |-- ...
|   |-- dataset_six.csv
|-- processed_data/
|   |-- combined_cleaned_dataset.csv
|-- scripts/
|   |-- data_cleaning.py
|   |-- regression_analysis.py
|   |-- classification_analysis.py
|   |-- clustering_analysis.py
|-- README.md
