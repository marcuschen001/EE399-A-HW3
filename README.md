# EE399-A-HW3

## Data Classification
Marcus Chen, April 21, 2023

The classification or stratification of data is a supervised machine learning method, where a model trained with data and a target is used to predict a target with a specified set of testing data. In this project, we will use the MNIST dataset (a dataset filled with handwritten digits) to observe the qualities of multiple processes for data classification: linear classification (LDA), decision trees, and support vector machines (SVM).

### Introduction and Overview:
Data classification or data stratification is the process within supervised machine learning of separating data out into subsections, based on a designated assignment. Instead of just guessing correlation like in the last project, each data point has a designated grouping, or “target” that it is assigned to. For example, in a data classification model that classifies dogs and cats, the data would be an image and the target would be a corresponding string that read ‘dog’ or ‘cat.’ 

https://www.openml.org/search?type=data&sort=runs&id=554&status=active

In this project, the MNIST database is used to explore classification. The MNIST dataset is a series of 70,000 images of handwritten digits that are 28 × 28 pixels ranging from intensity levels 0-255, with a corresponding target digit. To begin with, an SVD analysis is performed with the first part of the data to find the perfect rank and the V nodes. 

Once the data is processed and projected onto PCA space, we can observe the effects of a linear classifier (LDA). For the first experiment, we will use the LDA over two digits and analyze the accuracy of the model; we will then use the LDA over three digits and analyze the same accuracy. Over all possible two-digit combinations, we will look for the pairs that are easiest to separate and the hardest to separate. 

In the next experiment, we will use a support vector machine (SVM) and a decision tree to see their success in separating all ten of the digits. Lastly, we will compare the LDA, decision tree, and SVD models over the pairs of digits that are easiest to separate according to the LDA and the hardest. 

### Theoretical Background: 
#### Singular value decomposition (SVD): 
Singular Value Decomposition (SVD) is the factorization of a 2D matrix of m × n into 3 component parts: a unitary matrix U that is m × m, a diagonal matrix that is scaled by a factor of the 1D matrix S, and the transpose of unitary n × n matrix V. The transpose of V is the principal component direction of the data.

$A_{m \times n} = U_{m \times m} \Sigma V_{n \times n}^T$

#### Linear Classifier (LDA):
A classification model based on the linear combination of the characteristics, or feature values. Features are presented to the machine in the form of a feature vector, denoted by this formula:

$y = f(\vec{w} \cdot \vec{x}) = f(\Sigma_j w_j x_j)$

#### Support Vector Machine (SVM):
A classification model that is an extension of the non-probabilistic linear classifier, that can also perform non-linear classification through the use of the kernel trick. SVM’s map points to space to try to separate categories as much as possible in training data in order to create gaps in which to classify new training and testing data. 

#### Decision Tree: 
A classification model that uses a hierarchical model of decisions and their possible consequences: in our case, the classification of data. Decisions are based on various quantitative values and are used to try to distinguish between data points that are otherwise deemed similar. A tree can be visualized as a piecewise constant approximation.

### Algorithm Interpretation and Development:

#### SVM: 
Because of the quantity of images within the dataset, using scipy’s SVD function alone will result in an overflow error. To remedy those issues, we are only using the first 20,000 parts of the data.

```
X_col = X[:20000].T
U, S, Vh = np.linalg.svd(X_col)
```

In order to find the minimum ranking r, the percentage of variance of all the SVD nodes, S, is used. A while loop is used to iterate through the nodes until the cumulative value reaches the SVD node defined by an arbitrary threshold percentage 90% as shown here:

```
target = 0.9
current = 0.0
i = 0
while (current < target):
  current += S[i]**2 / np.sum(S**2)
  i += 1
```

#### PCA:
Based on the rank r, the PCA is performed by using the following method but setting n_components to r:

```
pca = PCA(n_components=52)
X_pca = pca.fit_transform(X)
```

#### Data Processing:
In order to prune the dataset so that it can be used to train and test classifiers, a simple logical-or function from numpy is used in order to only include the relevant numerical data:

```
X_cropped = X_pca[np.logical_or(y == 'digit0', y == 'digit1')]
y_cropped = y[np.logical_or(y == 'digit0', y == 'digit1')]
```

For data that desires three digits, one logical-or statement can be able to be put into the parameters of the other as so:

```
X_cropped = X_pca[np.logical_or(y == 'digit0', np.logical_or(y == 'digit1', y == 'digit2'))]
y_cropped = y[np.logical_or(y == 'digit0', np.logical_or(y == 'digit1', y == 'digit2'))]
```

In order to split the data into training and testing data, this function from scikit-learn’s model selection package is used, where 20% or 0.2 of the data is used for testing:

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

#### Classifiers:
For the three different types of classifiers, there are scikit-learn packets dedicated towards each of them. In order to perform an LDA, the linear_classifier package is used like so:

```
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

clf = LDA()
```

In order to perform an SVM, the SVC package from the svm package is used like so:

```
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
```

In order to perform a decision tree, the decision_tree package is used like so:

```
from sklearn import tree

clf = tree.DecisionTreeClassifier()
```

To fit any of them into the data, the packages have a fit method that can be used to adjust to the training data like so:

```
clf.fit(X_train, y_train)
```

To make a prediction, the packages have a prediction method that can be used and will return a testing data like so:

```
y_pred = clf.predict(X_test)
```

It is of note that for the decision tree classifier package, it only categorizes into two different sections and doing more would be difficult. 

### Computational Results:

![download (3)](https://user-images.githubusercontent.com/66970342/233757889-8bcb66d1-5f9f-40d6-90e2-07dbc12a8008.png)

This singular value spectrum was created by plotting out the variables of the S array, used to form the $\Sigma$ matrix. By performing the equation above, the minimum rank value that reaches our arbitrary image reconstruction threshhold variable 0.9, using the variance percentage is 

```
52
```

Within the SVD, the $\Sigma$ matrix is the variance of the principal components, the U matrix represents the component direction based on the rows, or the individual pixels in the images, and the V matrix represents the principal component direction based on the columns, or the images in the dataset. Some selections of the V matrix look like this as shown:

![download (6)](https://user-images.githubusercontent.com/66970342/233758141-49b7a703-9171-4aed-9035-890919a3f6e4.png)

By using the minimum rank, the data is simplified into something that can be used for classification. 

This is the results of an LDA that is trained on the digits 6 and 9:
![download (7)](https://user-images.githubusercontent.com/66970342/233758187-c2af9430-72de-49ef-aa49-d99e19d600e6.png)

The accuracy of the LDA is: 
```
0.9960245753523672
```

The results of an LDA trained on digits 1, 6, and 9:
![download (8)](https://user-images.githubusercontent.com/66970342/233758238-e96aa6d9-b36f-4d2b-b4de-e4566e23bf8f.png)

The accuracy of the LDA is:
```
0.9877964540640111
```

The accuracy of the three-digit LDA is possibly because of the slight increase in complexity of the data classification: going from 2 digits to 3 digits.

By looking at the LDA for every combination of digits, the accuracy is defined by:
```
0 vs. 1 Accuracy: 0.996617050067659
0 vs. 2 Accuracy: 0.983447283195394
0 vs. 3 Accuracy: 0.9928800284798861
0 vs. 4 Accuracy: 0.9959941733430444
0 vs. 5 Accuracy: 0.9818456883509834
0 vs. 6 Accuracy: 0.987300435413643
0 vs. 7 Accuracy: 0.995774647887324
0 vs. 8 Accuracy: 0.9839766933721777
0 vs. 9 Accuracy: 0.991345113595384
1 vs. 2 Accuracy: 0.9835238735709482
1 vs. 3 Accuracy: 0.987017310252996
1 vs. 4 Accuracy: 0.995239714382863
1 vs. 5 Accuracy: 0.9887244538407329
1 vs. 6 Accuracy: 0.9962724500169434
1 vs. 7 Accuracy: 0.990112063282795
1 vs. 8 Accuracy: 0.9625977558653519
1 vs. 9 Accuracy: 0.9925851027974385
2 vs. 3 Accuracy: 0.9667492041032897
2 vs. 4 Accuracy: 0.9790083242851972
2 vs. 5 Accuracy: 0.9635475385193536
2 vs. 6 Accuracy: 0.9722422494592646
2 vs. 7 Accuracy: 0.9772488624431221
2 vs. 8 Accuracy: 0.9612739775606225
2 vs. 9 Accuracy: 0.9813620071684588
3 vs. 4 Accuracy: 0.9899749373433584
3 vs. 5 Accuracy: 0.94908955778521
3 vs. 6 Accuracy: 0.9893009985734664
3 vs. 7 Accuracy: 0.9816418427433322
3 vs. 8 Accuracy: 0.9591982820329277
3 vs. 9 Accuracy: 0.9716312056737588
4 vs. 5 Accuracy: 0.9863013698630136
4 vs. 6 Accuracy: 0.9890510948905109
4 vs. 7 Accuracy: 0.9847733711048159
4 vs. 8 Accuracy: 0.9893772893772894
4 vs. 9 Accuracy: 0.9513964454116793
5 vs. 6 Accuracy: 0.9704321455648218
5 vs. 7 Accuracy: 0.9893460690668626
5 vs. 8 Accuracy: 0.9509132420091324
5 vs. 9 Accuracy: 0.975894538606403
6 vs. 7 Accuracy: 0.9971771347918137
6 vs. 8 Accuracy: 0.9850419554906968
6 vs. 9 Accuracy: 0.9949403686302855
7 vs. 8 Accuracy: 0.9865439093484419
7 vs. 9 Accuracy: 0.9505436688881095
8 vs. 9 Accuracy: 0.9756982227058397
```

Out of the list, the digits that are "easiest" to compare are 6 and 7 and the digits "hardest" to compare are 3 and 5. 

Using the SVM and the decision tree, these classifications happen:
![download (9)](https://user-images.githubusercontent.com/66970342/233758389-69eaa532-531d-458a-9482-c700aad3ced9.png)
![download (10)](https://user-images.githubusercontent.com/66970342/233758398-9cb088bb-5c3e-4aee-a422-90fb536baecc.png)

The accuracies of the SVM and decision trees are:
```
SVM Accuracy: 0.9831428571428571
Decision Tree Accuracy: 0.19235714285714287
```

The inaccuracies of the decision tree can be characterized by the binary limitations of the scikit-learn package for the decision tree; for only being able to judge 2 numbers, it was decent.

Comparing the LDA, SVM, and decision tree for the easiest digit pair to distinguish these are the results:
![download (11)](https://user-images.githubusercontent.com/66970342/233758504-1632d1a7-477b-48a3-a2c9-1763d39ba225.png)
![download (12)](https://user-images.githubusercontent.com/66970342/233758508-129b6d8c-44a7-48cf-bcf6-b32c7bcc9383.png)
![download (13)](https://user-images.githubusercontent.com/66970342/233758510-581985a4-ccde-4e5b-ac7b-3b6a9f8e3697.png)

The accuracies are:
```
LDA Accuracy: 0.9982357092448836
SVM Accuracy: 1.0
Decision Tree Accuracy: 0.9904728299223712
```

With the hardest pair of digits to distinguish these are the results:
![download (14)](https://user-images.githubusercontent.com/66970342/233758582-da07ead0-0ad9-458b-a93b-e1b46aa0d018.png)
![download (15)](https://user-images.githubusercontent.com/66970342/233758587-1831f573-2c38-4637-845f-fb63308e41a7.png)
![download (16)](https://user-images.githubusercontent.com/66970342/233758591-acdcc59b-3e47-4bb1-a6e4-0cece91a3999.png)

The accuracies are:
```
LDA Accuracy: 0.9435154217762913
SVM Accuracy: 0.9903381642512077
Decision Tree Accuracy: 0.9208472686733556
```

### Conclusions:

As a comparison, it was easy to see that out of all the classification methods used, that support vector machines were the most accurate when it came to testing data, but all three of them relatively scored above 0.9 on accuracy within a binary classification scheme. While decision trees can be able to do complex classification of more than two classifications, for the case of Python it is more complicated than what is given in scikit-learn. 

In comparison to unsupervised learning, which tries to classify data blindly, supervised data is useful in that a machine is already given a classification with the targets. Up until 2014, classification models such as SVM and decision trees were state-of-the-art for the use of AI classification and prediction for that reason; since then, a hybrid model that combines components of supervised and unsupervised learning has become more popular. 
