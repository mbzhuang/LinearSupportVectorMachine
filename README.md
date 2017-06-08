# Linear Support Vector Machine With the Squared Hinge Loss

- Statistical Machine Learning For Data Scientists Code Release Practice

- In this repo, I implemented in Python an algorithm of linear support vector machine with the squared hinge loss.

<p align="center">
The loss function for linear support vector machine with the squared hinge loss is
</p>

![Alt text](www/eq1.gif?style=centerme)

<p align="center">
Thus the gradient of the loss function can be written as
</p>
<p align="center"> 
<img src="www/eq2.gif">
</p>
<p align="center">
![Alt text](?raw=true "Title")
</p>
<p align="center">
![Alt text](www/eq3.gif?raw=true "Title")
</p>


**There are two demos and one function py file in this repo:**

**SVM_spam_data_demo.ipynb**

Run this file in jupyter notebook, users can launch the method on spam dataset, visualize the training process, and print the performance.
-	Spam dataset: https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data
-	Test_indicator: https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest

**SVM_simulated_data_demo.ipynb**

Run this file in jupyter notebook, users can launch the method on a simple simulated dataset,
visualize the training process, and print the performance.

**svm_hingeloss.py**

Users can run an experimental comparison between my implementation and scikit-learn’s on either a simulated or real-world dataset of their choice.

