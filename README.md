# Principal Component Analysis (PCA)

A practical and conceptual study of **Principal Component Analysis (PCA)** covering dimensionality reduction, explained variance, scree plots, and a hands-on Python implementation using the **Glass Classification Dataset**.

This repository is designed to connect the **mathematical intuition behind PCA** with its practical implementation using Python and Scikit-learn.

---

## 📌 Project Overview

Modern datasets may contain dozens, hundreds, or even thousands of features. Although additional features can contain useful information, high-dimensional datasets can introduce several challenges, including:

* Increased computational complexity
* Longer model training time
* Difficulty in visualizing the data
* Redundant or correlated features
* Potential overfitting
* Increased storage requirements

**Principal Component Analysis (PCA)** is a dimensionality-reduction technique that transforms the original features into a smaller number of new variables called **Principal Components**.

These components are constructed so that the first Principal Component captures the largest possible amount of variance, followed by the second component, third component, and so on.

---

## 🧠 Topics Covered

This repository covers both the theoretical and practical aspects of PCA.

### 1. Curse of Dimensionality

As the number of features increases, working with the data becomes more computationally expensive and difficult to visualize.

The project discusses why high-dimensional data can create challenges for Machine Learning algorithms and how dimensionality reduction can help.

### 2. Feature Selection vs. Feature Extraction

Two common approaches for reducing dimensionality are:

**Feature Selection**

* Selects a subset of the existing features.
* Original variables remain unchanged.

**Feature Extraction**

* Creates new features from combinations of the existing variables.
* PCA is a feature-extraction technique.

---

## 🔍 What is PCA?

**Principal Component Analysis (PCA)** transforms data from its original feature space into a new coordinate system.

The new dimensions are known as:

* Principal Component 1 — PC1
* Principal Component 2 — PC2
* Principal Component 3 — PC3
* ...

PC1 captures the greatest amount of variance in the dataset.

PC2 captures the next greatest amount of variance while remaining orthogonal to PC1.

The process continues for the remaining components.

Each Principal Component is a **linear combination of the original features**.

---

## ⚙️ PCA Workflow

The conceptual PCA workflow explored in this project is:

### Step 1 — Standardize the Data

Because PCA is influenced by feature scale, the numerical features are standardized before PCA is applied.

Using `StandardScaler`, features are transformed approximately to:

* Mean = 0
* Standard deviation = 1

### Step 2 — Analyze Covariance

The covariance structure helps describe how variables vary together.

PCA uses these relationships to identify the directions containing the greatest variation in the dataset.

### Step 3 — Determine Eigenvalues and Eigenvectors

Conceptually:

* **Eigenvectors** represent the directions of the Principal Components.
* **Eigenvalues** indicate how much variance is associated with those directions.

Components with larger eigenvalues contain more information about the variation in the data.

### Step 4 — Calculate Explained Variance

The **Explained Variance Ratio (EVR)** indicates the proportion of the dataset's variance captured by each Principal Component.

Cumulative explained variance can then be used to determine how many components should be retained.

### Step 5 — Select the Number of Components

A Scree Plot or cumulative explained-variance curve helps identify a suitable number of Principal Components.

The objective is to retain a substantial proportion of the dataset's information while reducing dimensionality.

---

# 📊 Dataset

The practical implementation uses the **Glass Classification Dataset**.

The original dataset contains:

* **214 observations**
* **9 numerical input features**
* A target variable representing the glass class

### Input Features

| Feature | Description      |
| ------- | ---------------- |
| RI      | Refractive Index |
| Na      | Sodium           |
| Mg      | Magnesium        |
| Al      | Aluminium        |
| Si      | Silicon          |
| K       | Potassium        |
| Ca      | Calcium          |
| Ba      | Barium           |
| Fe      | Iron             |

The dataset contains six observed target classes:

`1, 2, 3, 5, 6, 7`

---

## 🧹 Data Preprocessing

Before PCA, several preprocessing steps are performed.

### Remove Unnecessary Index Column

The dataset contains an `index` column that is not required as a predictive feature, so it is removed.

### Missing Value Analysis

The dataset is checked for missing values.

No missing values were identified in the provided data.

### Duplicate Analysis

One duplicate observation was identified and removed before continuing with the analysis.

### Separate Features and Target

The dataset is divided into:

* `X` → Input features
* `y` → Target/Class

PCA is applied to the input features rather than the target variable.

---

## 📏 Feature Standardization

The nine input features operate on different numerical scales.

Therefore, the project uses:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
```

Standardization is particularly important for PCA because variables with larger numerical scales could otherwise have a disproportionate influence on the calculated components.

---

# 🔬 Applying PCA

PCA is first fitted without specifying the number of components:

```python
from sklearn.decomposition import PCA

pca = PCA()
principal_components = pca.fit_transform(scaled_data)
```

This allows the explained variance associated with all available components to be examined.

---

## 📈 Explained Variance Analysis

Cumulative explained variance is visualized using:

```python
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance")
plt.show()
```

For the processed Glass dataset, the approximate cumulative explained variance is:

| Principal Components | Cumulative Variance |
| -------------------: | ------------------: |
|                    1 |               27.9% |
|                    2 |               50.8% |
|                    3 |               66.4% |
|                    4 |               79.1% |
|                    5 |               89.3% |
|                    6 |               95.2% |
|                    7 |               99.3% |
|                    8 |              99.98% |
|                    9 |                100% |

This demonstrates the trade-off between **dimensionality reduction** and **information retention**.

For example:

* 5 components retain approximately **89%** of the variance.
* 6 components retain approximately **95%** of the variance.

Therefore, the appropriate number of components depends on the acceptable information-retention threshold for the application.

---

## 🔄 Dimensionality Reduction

The original feature space contains:

**9 input dimensions**

After PCA, the dataset can be represented using fewer Principal Components.

For example:

```python
pca = PCA(n_components=6)
reduced_data = pca.fit_transform(scaled_data)
```

This transforms:

**9 Original Features → 6 Principal Components**

while retaining approximately **95% of the total variance** in this dataset.

A more aggressive reduction to 5 components retains approximately **89%**.

---

# 📊 Scree Plot / Cumulative Variance Plot

The explained-variance plot is an important part of PCA because it helps determine how much information is retained as additional Principal Components are included.

The point where adding further components provides relatively small gains can be used as one criterion when choosing dimensionality.

However, the final number of components should also depend on:

* Required information retention
* Model performance
* Computational constraints
* Interpretability requirements
* Downstream task

---

# ✅ Advantages of PCA

PCA can provide several benefits:

* Reduces the dimensionality of datasets
* Reduces redundant information
* Handles correlated input variables
* Can decrease model training time
* Reduces computational requirements
* Supports data compression
* Can make high-dimensional data easier to visualize
* May reduce overfitting in some situations
* Can help reduce the influence of low-variance directions/noise

---

# ⚠️ Limitations of PCA

PCA also has several limitations:

* Principal Components are less interpretable than original variables
* Information can be lost when components are removed
* PCA captures linear relationships
* Results are sensitive to feature scaling
* Variance does not always equal predictive importance
* Selecting too few components may remove useful information

Therefore, PCA should not be applied automatically without evaluating the characteristics of the dataset and the goal of the Machine Learning task.

---

# 📂 Repository Files

```text
PCA/
│
├── PCA_Worked.ipynb
│   └── Complete Python implementation of PCA
│
├── PCA_notes.pdf
│   └── Conceptual notes covering PCA theory and workflow
│
├── glass.data
│   └── Dataset used for practical implementation
│
├── scree.PNG
│   └── Explained-variance / PCA visualization
│
└── README.md
    └── Project documentation
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# 💡 Key Learning Outcomes

Through this project, I explored:

1. Why high-dimensional data can become problematic.
2. The difference between feature selection and feature extraction.
3. The fundamental intuition behind PCA.
4. Why feature standardization is important before PCA.
5. The concepts of covariance, eigenvectors, and eigenvalues.
6. How Principal Components are generated.
7. How Explained Variance Ratio measures retained information.
8. How Scree/Cumulative Variance Plots support component selection.
9. How to implement PCA using Scikit-learn.
10. How to evaluate the trade-off between dimensionality reduction and information retention.

---

# 🔮 Possible Next Steps

Future extensions of this project can include:

* Comparing classification performance before and after PCA
* Comparing training time with original vs. reduced features
* Testing different explained-variance thresholds
* Visualizing samples using PC1 and PC2
* Examining PCA component loadings
* Comparing PCA with alternative dimensionality-reduction methods
* Evaluating PCA using different Machine Learning classifiers

---

# 🎯 Conclusion

Principal Component Analysis is a powerful technique for transforming a high-dimensional dataset into a lower-dimensional representation while retaining as much meaningful variance as possible.

This implementation demonstrates the complete process from **data preprocessing and standardization to explained-variance analysis and dimensionality reduction**.

The Glass dataset begins with **9 numerical input features**. PCA shows that approximately **95.2% of its variance can be represented using the first 6 Principal Components**, demonstrating how PCA can reduce dimensionality while preserving most of the variation present in the original data.

---

## 🤝 Connect

If you find this project useful, feel free to:

⭐ Star the repository
🍴 Fork the project
💡 Suggest improvements
🤝 Connect for discussions around Machine Learning, AI, and Data Science

**Arshad Abbas | AI Researcher**
