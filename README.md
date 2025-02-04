# Machine Learning A-Z: AI & ipynbthon [2025] - Course Codes

This repository contains the code implementations from the **Machine Learning A-Z: AI & ipynbthon [2025]** course on Udemy, led by **Kirill Eremenko** and **Hadelin de Ponteves, Hon. PhD**. The course provides hands-on experience in machine learning, deep learning, and artificial intelligence using **ipynbthon**, **TensorFlow**, and **Keras**.

## Course Overview

The course covers a wide range of machine learning topics, from fundamental concepts to advanced techniques, including:

- **Supervised Learning**  
  Learn how to implement algorithms for classification and regression tasks, such as linear regression, decision trees, and random forests.

- **Unsupervised Learning**  
  Dive into clustering techniques like K-means and hierarchical clustering, and dimensionality reduction methods such as PCA, Kernel PCA, and LDA.

- **Natural Language Processing (NLP)**  
  Explore text data manipulation with **NLTK**, focusing on tasks like tokenization, stemming, and sentiment analysis.

- **Computer Vision**  
  Build image classifiers and object detection models using **TensorFlow** and **Keras**.

- **Model Selection & Optimization**  
  Learn how to tune hyperparameters, evaluate model performance with cross-validation, and optimize models for real-world applications.

## Folder Structure

Each folder in this repository corresponds to a specific section of the course and contains the code related to that topic. The structure is as follows:

```
Machine-Learning-A-Z-Course-Codes/
│
├── Data-Preprocessing-Template/
|   └──data_preprocessing_.ipynb
|
├── Supervised-Learning/
│   └── Regression/
|   |    |──simple_linear_regression.ipynb
|   |    |── multiple_linear_regression.ipynb
|   |    |── polynomial_regression.ipynb
|   |    |── support_vector_regression.ipynb
│   |    ├── decision_tree_regression.ipynb
│   |    |── random_forest_regression.ipynb
|   |    └── r2_score_regression.ipynb   
|   |
│   └── Classification/
|   |    ├── logistic_regression.ipynb
|   |    |── knn_classification.ipynb
|   |    |── svm_classification.ipynb
|   |    |── kernel_svm_classification.ipynb
│   |    ├── naive_bayes_classification.ipynb
│   |    |── random_forest_classification.ipynb
|   |    └── decesion_tree_classification.ipynb
|   |
|   └── XGBoost/
|        |── xg_boost.ipynb
│
├── Unsupervised-Learning/
│   ├── k_means.ipynb
│   └── hierarchical_clustering.ipynb
│
├── Reinforcement-Learning/
│   ├── thomspon_sampling.ipynb
│   └── upper_confidence_bound.ipynb
|
├── Association-Rule-Learning/
│   ├── apriori.ipynb
│   └── eclat.ipynb
|
├── NLP/
│   └── sentiment_analysis.ipynb
│
├── Deep Learning/
│   ├── artificial_neural_network.ipynb
│   └── convolutional_neural_network.ipynb
│
├── Model-Optimization/
│   ├── grid_search.ipynb
│   └── k_fold_cross_validation.ipynb
│
└── Dimensionality-Reduction/
    ├── pca.ipynb
    ├── kernel_pca.ipynb
    └── lda.ipynb
```

## Installation

To run the code locally, you'll need ipynbthon installed on your machine. You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

The **requirements.txt** file includes essential libraries like:

- `numipynb`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `keras`
- `matplotlib`
- `seaborn`

## Usage

Each ipynbthon script in this repository corresponds to a specific lesson or concept in the course. You can run the scripts directly to see the models in action and learn how to implement machine learning techniques in ipynbthon.

For example, to run a linear regression model:

```bash
ipynbthon Supervised-Learning/linear_regression.ipynb
```

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request. Contributions are always welcome!

## License

This repository is provided under the **MIT License**. You are free to use, modify, and distribute the code as per the terms of the license.

## Acknowledgments

A special thanks to **Kirill Eremenko**, **Hadelin de Ponteves**, and the **SuperDataScience Team** for their excellent teaching and resources. This course provided a solid foundation for understanding and implementing machine learning algorithms.
