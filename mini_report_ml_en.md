## Mini-report – Machine Learning Models (Supervised, Unsupervised, Reinforcement)

### Abstract
Machine learning (ML) refers to methods that learn from data to predict quantities, assign classes, discover structure, or support decisions. This mini-report focuses on **supervised learning**, and briefly outlines **unsupervised learning** and **reinforcement learning**.

### 1. Introduction: what is an ML “model”?
An ML model is a parameterized function $f_\theta$ that maps an input vector of features $\mathbf{x}$ to an output $\hat{y}$. In regression, $\hat{y}$ is real-valued; in classification, $\hat{y}$ is a label or a probability. Learning selects parameters $\theta$ that minimize a loss function $\mathcal{L}$ measuring the discrepancy between $\hat{y}$ and ground truth $y$, with the key goal of **generalization** to unseen data.

### 2. Supervised Machine Learning
#### 2.1. Data, objective, and generalization
In supervised learning, the dataset is composed of labeled pairs $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$. The feature vector $\mathbf{x}$ aggregates explanatory variables (sensor readings, operational settings, geological descriptors, and so on), while $y$ is the target variable to predict. The learning task is to infer a mapping from inputs to targets that remains accurate beyond the training examples, which requires both sound methodology and careful evaluation.

#### 2.2. Modeling workflow (pipeline)
A typical supervised workflow is to define the target and constraints, prepare the data (cleaning, missing values, encoding), and split the dataset to measure generalization. Models are usually built from simple baselines to more expressive approaches, and compared using consistent metrics.

One of the most common pitfalls is **data leakage**: using information that would not be available at prediction time (or that indirectly encodes the label). Leakage produces overly optimistic evaluation scores and leads to poor real-world performance.

#### 2.3. Train/validation/test split
Data splitting organizes evaluation. The training set is used to fit model parameters $\theta$, the validation set is used to select models and tune hyperparameters without touching the test set, and the test set provides the final unbiased estimate. For time-dependent data, shuffling across time can accidentally mix past and future; chronological splits are usually preferred to preserve realism.

#### 2.4. Loss functions and learning
Training typically solves an optimization problem of the form $\min_\theta \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$. In regression, common losses include MSE (more sensitive to large errors) and MAE (often more robust). Optimization depends on the model family (e.g., gradient-based methods for neural networks, iterative solvers for SVMs).

#### 2.5. Major families of supervised models
Common choices include **linear models** (fast and interpretable), **tree-based ensembles** (Random Forest, Gradient Boosting, often strong on tabular data), **SVM/SVR** (effective in high-dimensional settings but sometimes costly), and **neural networks** (very flexible for images, text, and sequences, but typically requiring more tuning). The right choice depends on data size, nonlinearity, and interpretability needs.

#### 2.6. Evaluation: $R^2$, $R$, MSE, RMSE, and MAE
In regression, error magnitude is summarized with **MSE** and **MAE**, while **RMSE** ($\sqrt{\mathrm{MSE}}$) expresses error in the same unit as the target. Goodness of fit is often reported with $R^2 = 1-\frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}$ (comparison to a mean predictor) and the correlation coefficient $R$ between $y$ and $\hat{y}$, which indicates how well predicted variations track true variations.

#### 2.7. Overfitting, bias–variance, and regularization
Underfitting happens when the model is too simple; overfitting happens when it is too flexible and generalizes poorly. Regularization (Ridge/Lasso, early stopping, tree constraints) and cross-validation help improve generalization.

#### 2.8. Features, scaling, and robustness
On tabular problems, feature design often dominates performance; transformations, aggregations, and interactions can help. Scaling is essential for distance- or magnitude-sensitive models (kNN, SVM, regularized linear models) and less critical for tree-based models.

#### 2.9. Interpretability and uncertainty
Interpretability can be global (feature importance, average effects) or local (explaining a single prediction). Tools such as SHAP are commonly used; depending on the use case, uncertainty can be quantified with prediction intervals or calibration.

### 3. Unsupervised Machine Learning
Unsupervised learning has no target label and focuses on structure discovery through clustering, dimensionality reduction, and anomaly detection. Evaluation is often indirect and relies on stability checks and domain expertise.

### 4. Reinforcement Learning
RL learns a policy for sequential decisions: an agent observes $s_t$, chooses $a_t$, and receives rewards, aiming to maximize a discounted return $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$. RL is powerful for control tasks but is more demanding in practice (exploration/exploitation, large interaction needs, safety constraints).

### 5. Conclusion
Supervised ML is usually the most direct framework for predicting a target from labeled data. Unsupervised learning complements it by uncovering structure and anomalies, while RL is suitable for sequential decision-making when long-term objectives must be optimized under constraints.
