# Data Science Workshop - NBA Free Throws Prediction

In this repository we present a project as part of a Data Science Workshop at Tel-Aviv University.

Our main purpose was to complete an entire end to end Data Science pipeline start from data collecting through data
cleaning and exploring, until the modeling and predictions. 

As we will explain later, we found out that since a free throw its a statistical event, machine learning models having troubles to predict its outcome with high accuracy. 

In addition, since our data was collected from pro NBA players performances, it consisted of 76% made shots, caused our data set to be an imbalance one. 

As a result all of our primary modelstend to label all shots as scores. Therefore, our point offocus in this work was dealing with that phenomenon and try increasing the recall parameter on the missed shots.