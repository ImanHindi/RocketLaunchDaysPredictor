# Pandas library is used for handling tabular data
from email.mime import image
from random import seed
import pandas as pd

# NumPy is used for handling numerical series operations (addition, multiplication, and ...)

import numpy as np
# Sklearn library contains all the machine learning packages we need to digest and extract patterns from the data
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split

# Machine learning libraries used to build a decision tree
from sklearn.tree import DecisionTreeClassifier

# Sklearn's preprocessing library is used for processing and cleaning the data 
from sklearn import preprocessing

# for visualizing the tree
import pydotplus
from IPython.display import Image

#import Data:
launch_data=pd.read_excel('RocketLaunchDataCompleted.xlsx')
print(launch_data.head())

print(launch_data.columns)
print(launch_data.info())

#CleanData:
launch_data['Launched?'].fillna('N',inplace=True)
launch_data['Crewed or Uncrewed'].fillna('Uncrewed',inplace=True)
launch_data['Wind Direction'].fillna('unKnown',inplace=True)
launch_data['Condition'].fillna('Fair',inplace=True)
launch_data.fillna(0,inplace=True)
print(launch_data.head())
print(launch_data.info())

#DataManipulation:
label_endcoder=preprocessing.LabelEncoder()
launch_data['Launched?']=label_endcoder.fit_transform(launch_data['Launched?'])
launch_data['Crewed or Uncrewed']=label_endcoder.fit_transform(launch_data['Crewed or Uncrewed'])
launch_data['Condition']=label_endcoder.fit_transform(launch_data['Condition'])
launch_data['Wind Direction']=label_endcoder.fit_transform(launch_data['Wind Direction'])


print(launch_data.head())





# First, we save the output we are interested in. In this case, "launch" yes and no's go into the output variable.
y = launch_data['Launched?']

# Removing the columns we are not interested in
launch_data.drop(['Name','Date','Time (East Coast)','Location','Launched?','Hist Ave Sea Level Pressure','Sea Level Pressure','Day Length','Notes','Hist Ave Visibility', 'Hist Ave Max Wind Speed'],axis=1, inplace=True)

# Saving the rest of the data as input data
X = launch_data




print(X.columns)


#classifier=DecisionTreeClassifier(max_depth=5,random_state=0)
tree_model = DecisionTreeClassifier(random_state=0,max_depth=5)


#x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=99)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=99)

#classifier.fit(x_train,y_train)
tree_model.fit(X_train,y_train)
print(tree_model)
#y_pred=classifier.predict(x_test)
y_pred = tree_model.predict(X_test)
result=(y_test,y_pred)
print(result)

score=tree_model.score(X_test,y_test)

print(score)

# ['Crewed or Uncrewed', 'High Temp', 'Low Temp', 'Ave Temp',
#        'Temp at Launch Time', 'Hist High Temp', 'Hist Low Temp',
#        'Hist Ave Temp', 'Precipitation at Launch Time',
#        'Hist Ave Precipitation', 'Wind Direction', 'Max Wind Speed',
#        'Visibility', 'Wind Speed at Launch Time', 'Hist Ave Max Wind Speed',
#        'Hist Ave Visibility', 'Condition']

data_input = [ 1.  , 75.  , 68.  , 71.  ,  0.  , 75.  , 55.  , 65.  ,  0.  , 0.08,  0.  , 16.  , 15.  ,  0.  ,  0. ]

y_pred=tree_model.predict([data_input])
print(y_pred)

import sklearn.tree as tree

from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt
def tree_graph_to_png(model_tree,feature_names,class_names,png_file_to_save):

    plt.figure(figsize=(20, 10))
    tree.plot_tree(model_tree, 
                   filled=True, rounded=True, 
                   feature_names = X_train.columns,
                   class_names = class_names,
                   fontsize=12)
    plt.show()
    tree_str = tree.export_graphviz(model_tree, feature_names=feature_names, class_names=class_names,
                                     filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)   
    Image(graph.create_png())

    
    return Image(graph.create_png())


y=tree_graph_to_png(model_tree=tree_model, feature_names=X.columns.values,class_names=['No Launch','Launch'], png_file_to_save='decision-tree.png')




