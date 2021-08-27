# DecisionTree-HeartDisease
In this project, I have implemented a **Decision Tree** to predict if someone is suffering from heart disease or no based on sex, age, blood pressure and other criterias, using *Python* and *Sickit-learn*. My dataset is retreived from <a href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease">*UCLI Machine Learning Repository*</a>. Following steps show the implementation process in detail:

* Preprocessing: One-Hot Encoding and filling or removing missing data
* Seperating train data and test data
* Generating first dicision tree
* Evaluating the tree using **Confusion Matrix**
* Pruning the tree using **Cost Complexity Pruning**
* Calculating accuracy for different alpha values
* Executing **5-fold Cross Validation** and calculating Mean, Variance and Accuracy to find the best alpha
* Regenerating dicision tree based on best obtained alpha
* Drawing Confusion Matrix again
* Showing final dicision tree

By running the code using following command, you can see the results by yourself:

```
    python src.py
```

But before that you need to have the following Python packages installed:
* pandas >= 0.25.1
* numpy >= 1.17.2
* sklearn >= 0.22.1

A very complete **Persian Report** is also included in Report.pdf. 

Final Decision Tree:

<p align="center">
  <img src="https://github.com/ghazalb76/DecisionTree-HeartDisease/blob/main/resultPics/Capture7.PNG">
</p>


