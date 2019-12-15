# Cancer_Predictor_App

The application was deployed and available online: [lead me to the app](https://cancer-predictor-app.herokuapp.com).

The app is to predict whether a tumor of breast cancer is benign or maglinant. Users will input required information in order to make predictions. 

## Inputs: ##
* Age: a number
* Shape: 1 - 4
* Margin: 1 - 5
* Density: 1 - 4
* Androgen history: yes or no
* Previous visit: yes or no
* Blood group: either A, B, O, or AB

Data can be downloaded on [Kaggle](https://www.kaggle.com/overratedgman/mammographic-mass-data-set) or contact me for the dataset. A variety of Machine Learning algorithms was applied on the dataset:

* Support Vector Machine
* Decision Tree
* Random Forest
* XG Boost
* Naive Bayes
* Nearest Neighbor
* Logistic Regression

Learning curves of the models and a table for comparison purpose were added below. The final model was built with Decision 
tree, which was the best classifier for the dataset with 90% accuracy obtained. 

## Dependencies
* Flask==1.1.1
* gunicorn==19.9.0
* itsdangerous==1.1.0
* Jinja2==2.10.1
* MarkupSafe==1.1.1
* Werkzeug==0.15.5
* numpy>=1.9.2
* scipy>=0.15.1
* scikit-learn.=0.18
* matplotlib>=1.4.3
* pandas>=0.19

## Model comparison:

### Support Vector Machine
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/SVM.png" width="750" height="250">

### Decision Tree
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/DT.png" width="500" height="500">

### Random Forest
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/RF.png" width="500" height="500">

### XG Boost
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/XGB.png" width="500" height="500">

### Naive Bayes
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/NB.png" width="500" height="500">

### Nearest Neighbor
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/NN.png" width="500" height="500">

### Logistic Regression
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/LR.png" width="800" height="300">
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/LR2.png" width="500" height="500">

### Table for comparison
<img src="https://raw.githubusercontent.com/Trangle91/Cancer_Predictor_App/master/images/table.png" width="900" height="400">

## Limitations
Due to short computational power, GridSearchCV ran on top of SVM was not completely implemented, the accuracy above was the result of the default parameters of SVM. A CNN model could be built for further accuracy improvement.
