# CICIDS2017 Data Analysis

## Objective
In this project, I analyze the CICIDS2017 dataset to find meaningful insights using exploratory data analysis (EDA), machine learning (ML), and deep learning (DL) techniques. The project is part of an assignment for a Data Scientist role at Sleek.

---

## Data
The CICIDS2017 dataset is used for evaluating network intrusion detection systems (IDS) and includes both benign and malicious network traffic data.

- **Dataset Size**: 2.2 million rows with 79 features.
- **Target Variable**: The 'Label' column indicates whether the traffic is benign or represents an attack.

---

## Setup

Here’s how you can set up the environment and run the code:

### 1. Clone the repository:
```bash
git clone https://github.com/ShaiCh1/CICIDS2017_project.git
```

### 2. Navigate to the project folder:
```bash
cd CICIDS2017_project
```

### 3. Install required libraries:
Use the `requirements.txt` file to install dependencies.
```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation:
Due to file size limitations on GitHub, the dataset files are not included in this repository. Please download the **CICIDS2017 dataset** from the official source and place it in the project directory. 

You can download the dataset CICIDS2017 from the official site.

Once downloaded, follow these steps:
1. Place the dataset in the project folder (as `dataset.csv`).
2. The **EDA script** will clean and preprocess the data, generating two files:
   - **cleaned_data_before_pca.csv**: The cleaned dataset without PCA.
   - **cleaned_data_with_pca.csv**: The dataset after applying PCA.
   
   Choose one of these files for model training, depending on whether you want to use PCA or not.

---

This update ensures users are aware that they need to download the dataset themselves and that the generated CSV files will be used for training the models, providing flexibility with or without PCA.
---

### Running the Code

Once the setup is done, follow these steps to run the scripts:

1. **Exploratory Data Analysis (EDA)**:
   - For data exploration and preparation, run:
   ```bash
   python EDA.py
   ```

2. **Training the Models**:

   - **XGBoost**:
     ```bash
     python XG_boost.py
     ```

   - **Random Forest**:
     ```bash
     python Random_Forest.py
     ```

3. **Approximate Nearest Neighbors (ANN)**:
   - To implement and evaluate the Approximate Nearest Neighbors (ANN) algorithm for recommendation:
   ```bash
   python ANN.py
   ```

---

## Files

- **Requirements.txt**: List of required Python packages.
- **EDA.py**: Script for data exploration and preprocessing.
- **XG_boost.py**: Trains and evaluates the XGBoost model.
- **Random_Forest.py**: Trains and evaluates the Random Forest model.
- **ANN.py**: Performs recommendation using Approximate Nearest Neighbors (ANN).
- **cleaned_data_before_pca.csv**: Cleaned dataset (before PCA).
- **cleaned_data_with_pca.csv**: Dataset after PCA.
- **dataset.csv**: The original dataset file.

---

## Steps and Process

### 1. Data Preprocessing
   - **Missing Values**: Any infinite or missing values (around 3000 rows) were removed.
   - **Categorical Encoding**: Used Target Encoding for the 'Destination Port' column and label encoding for 'Label'.
   - **Scaling**: Standardized numerical features for PCA and model training.

### 2. Exploratory Data Analysis (EDA)
   - **Dataset Summary**: 2.2M rows and 79 columns with various network traffic features.
   - **Outliers**: Significant outliers were found in many features, but they were kept as the models (XGBoost, Random Forest) handle them well.
   - **Label Imbalance**: The 'Label' column was highly imbalanced, with benign traffic far outweighing attack traffic. This imbalance was considered when training the models.
   - **Correlation Matrix**: Identified correlations between features and the target label using a heatmap.

### 3. Dimensionality Reduction
   - **PCA**: Reduced the dimensionality of the dataset, retaining 25 components that explain 95% of the variance.

---

## Models

1. **XGBoost**:
   - **Without PCA**:
     - **Accuracy**: 99.91%
     - **Precision**: 91% for attack class 1 (which had very limited data), 100% for major classes.
     - **Recall**: 75% for class 1, 100% for major classes.
     - **F1-score**: High across all classes, with a drop only for class 1 due to limited data.
   - **With PCA**:
     - **Accuracy**: 99.83%
     - **Precision**: 95% for attack class 1, slightly lower than without PCA.
     - **Recall**: 47% for class 1, with some reduction in performance.
     - **F1-score**: Overall performance was reduced for class 1, while other classes performed well.

   - **Handling Class Imbalance**: XGBoost effectively handled the imbalanced dataset, maintaining high recall and precision across the majority of attack types except for class 1, which had very few data points.

2. **Random Forest**:
   - **Without PCA**:
     - **Accuracy**: 99.80%
     - **Precision**: 43% for attack class 1, with high precision for majority classes.
     - **Recall**: 93% for class 1, high recall for all other classes.
     - **F1-score**: Lower for class 1, but strong for the rest of the classes.
   - **With PCA**:
     - **Accuracy**: 99.86%
     - **Precision**: 80% for attack class 1, improved with PCA.
     - **Recall**: 63% for class 1, improving handling of this class.
     - **F1-score**: Balanced performance across the majority of classes, especially with PCA.

   - **Handling Class Imbalance**: Random Forest handled class imbalance well, especially with PCA, showing improved recall for class 1 and maintaining strong F1-scores for the other classes.

3. **Approximate Nearest Neighbors (ANN)**:
   - **Without PCA**:
     - **Accuracy**: 99.63%
     - **Precision, Recall, and F1-score**: All metrics were strong (99.63%).
     - **Average Distance**:
       - **Benign Traffic**: 1.2173
       - **Attack Traffic**: 0.0900
   - **With PCA**:
     - **Accuracy**: 99.31%
     - **Precision, Recall, and F1-score**: Consistently high metrics around 99.31%, even with the imbalanced dataset.
     - **Average Distance**:
       - **Benign Traffic**: 0.5862
       - **Attack Traffic**: 0.0488

   - **Handling Class Imbalance**: overall maintained strong precision, recall, and F1-scores for most classes.

---

## Insights

1. **Feature Importance and Performance**:
   - Both XGBoost and Random Forest achieved high accuracy (between **99.80% and 99.91%**). PCA slightly reduced accuracy, but the models still performed well, especially for classes with sufficient data.

2. **Class Imbalance**:
   - XGBoost and Random Forest handled the highly imbalanced data well, weighting the classes appropriately to ensure strong performance across the majority of classes. Class 1 (Bot - with limited data) showed reduced recall, but this is expected due to the small number of instances.

3. **Recommendation System (ANN Insight)**:
   - Using Approximate Nearest Neighbors (ANN), the recommendation system effectively distinguished between benign and attack traffic. The system achieved **99.31% accuracy** with PCA and **99.63% accuracy** without PCA. The shorter average distances for attack traffic indicate that attack patterns are more concentrated, making them easier to detect compared to benign traffic, which tends to be more spread out. The ANN model performed well despite the imbalanced data, showing high precision, recall, and F1-scores for identifying differences between benign and attack traffic.

---

## Conclusion

In summary:
   - XGBoost and Random Forest performed exceptionally well, even with imbalanced data. PCA was effective for reducing the dimensionality of the dataset while maintaining strong performance across most classes.
   - ANN provided useful insights into the distribution of benign versus attack traffic, achieving high accuracy and consistent precision, recall, and F1-scores across most classes, even with class imbalance.

---

## Final Notes

Future work could focus on:
   - Fine-tuning the models to improve recall for rare classes, possibly through data augmentation or balancing techniques like SMOTE.
   - Exploring alternative distance metrics in the ANN algorithm to better differentiate benign and attack traffic.
   - Investigating more advanced anomaly detection techniques to better capture rare attack types like those in class 1 (Bot).
