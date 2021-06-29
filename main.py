"""
Make the imports of python packages needed
"""
import tkinter as tk
from tkinter import ttk
from dpi_set import set_dpi_awareness
import pandas as pd
import numpy as np
from pprint import pprint
set_dpi_awareness()

# Import the dataset and define the feature as well as the target datasets / columns

dataset = pd.read_csv('dataset/diabetes.csv',
                      names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                             'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

print(dataset)
dataset = dataset.drop('Pregnancies', axis=1)
# We have dropped Pregnancy as it is not a major factor when considering the urban population.




#########################################################


"""
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
"""
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


#########################################################

#########################################################


"""
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
"""


def InfoGain(data, split_attribute_name, target_name="Outcome"):

    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    #Calculate the entropy of the dataset

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    # Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


#########################################################################


##########################################################################


"""
    ID3 Algorithm: This function takes five paramters:
    1. data = the data for which the ID3 algorithm should be run --> In the first run this equals the total dataset

    2. originaldata = This is the original dataset needed to calculate the mode target feature value of the original dataset
    in the case the dataset delivered by the first parameter is empty

    3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset --> Splitting at each node

    4. target_attribute_name = the name of the target attribute

    5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is
    also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
    space, we want to return the mode target feature value of the direct parent node.
"""


def ID3(data, originaldata, features, target_attribute_name="Outcome", parent_node_class=None):
    # Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#

    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable.

    elif len(features) == 0:
        return parent_node_class

    # If none of the above holds true, grow the tree!

    else:
        # Set the default value for this node --> The mode target feature value of the current node

        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]  # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        # gain in the first run
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature

        for value in np.unique(data[best_feature]):
            value = value
            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return (tree)

#################################################################################


#################################################################################


"""
    Prediction of a new/unseen query instance. This takes two parameters:
    1. The query instance as a dictionary of the shape {"feature_name":feature_value,...}

    2. The tree


    We do this also in a recursive manner. That is, we wander down the tree and check if we have reached a leaf or if we are still in a sub tree.
    Since this is a important step to understand, the single steps are extensively commented below.

    1.Check for every feature in the query instance if this feature is existing in the tree.keys() for the first call,
    tree.keys() only contains the value for the root node
    --> if this value is not existing, we can not make a prediction and have to
    return the default value which is the majority value of the target feature

    2. First of all we have to take care of a important fact: Since we train our model with a database A and then show our model
    a unseen query it may happen that the feature values of these query are not existing in our tree model because non of the
    training instances has had such a value for this specific feature.
    
    We can address this issue by giving the key to the nearest value possible.

    3. Address the key in the tree which fits the value for key --> Note that key == the features in the query.
    Because we want the tree to predict the value which is hidden under the key value (imagine you have a drawn tree model on
    the table in front of you and you have a query instance for which you want to predict the target feature
    - What are you doing? - Correct:
    You start at the root node and wander down the tree comparing your query to the node values. Hence you want to have the
    value which is hidden under the current node. If this is a leaf, perfect, otherwise you wander the tree deeper until you
    get to a leaf node.
    Though, you want to have this "something" [either leaf or sub_tree] which is hidden under the current node
    and hence we must address the node in the tree which == the key value from our query instance.
    This is done with tree[keys]. Next you want to run down the branch of this node which is equal to the value given "behind"
    the key value of your query instance 

    4. As said in the 2. step, we run down the tree along nodes and branches until we get to a leaf node.
    That is, if result = tree[key][query[key]] returns another tree object (we have represented this by a dict object -->
    that is if result is a dict object) we know that we have not arrived at a root node and have to run deeper the tree.
    Okay... Look at your drawn tree in front of you... what are you doing?...well, you run down the next branch...
    exactly as we have done it above with the slight difference that we already have passed a node and therewith
    have to run only a fraction of the tree --> You clever guy! That "fraction of the tree" is exactly what we have stored
    under 'result'.
    So we simply call our predict method using the same query instance (we do not have to drop any features from the query
    instance since for instance the feature for the root node will not be available in any of the deeper sub_trees and hence
    we will simply not find that feature) as well as the "reduced / sub_tree" stored in result.

    SUMMARIZED: If we have a query instance consisting of values for features, we take this features and check if the
    name of the root node is equal to one of the query features.
    If this is true, we run down the root node outgoing branch whose value equals the value of query feature == the root node.
    If we find at the end of this branch a leaf node (not a dict object) we return this value (this is our prediction).
    If we instead find another node (== sub_tree == dict objct) we search in our query for the feature which equals the value
    of that node. Next we look up the value of our query feature and run down the branch whose value is equal to the
    query[key] == query feature value. And as you can see this is exactly the recursion we talked about
    with the important fact that for each node we run down the tree, we check only the nodes and branches which are
    below this node and do not run the whole tree beginning at the root node
    --> This is why we re-call the classification function with 'result'
"""


def predict(query, tree):

    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                if key == "DiabetesPedigreeFunction":
                    temp_list = []
                    for t in tree[key].keys():
                        if t == key:
                            continue
                        else:
                            temp_list.append(float(t))
                    query[key] = str(closest(temp_list, float(query[key])))
                else:
                    temp_list = []
                    for t in tree[key].keys():
                        if t == key:
                            continue
                        else:
                            temp_list.append(int(t))
                    query[key] = str(closest(temp_list, int(query[key])))
                result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)

            else:
                return result


########################################################################

########################################################################


"""
Check the accuracy of our prediction.
The train_test_split function takes the dataset as parameter which should be divided into
a training and a testing set. The test function takes two parameters, which are the testing data as well as the tree model.
"""


def train_test_split(dataset):
    training_data = dataset.iloc[:700].reset_index(drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[700:].reset_index(drop=True)
    return training_data, testing_data


training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]


################################################################################

################################################################################


'''
    The clean functions actually takes the data values of the respective feature and rounds them off to nearest 10s or nearest 2nd decimal point
    This helps in increasing the accuracy. Moreover all the medical features are calculated in ranges thus it does not violate the validity of the 
    dataset 
'''


def clean_age(dataset, start, stop):
    for i in range(start, stop):
        if dataset.Age[i] is None:
            break
        temp=int(dataset.Age[i])
        dataset.Age[i]=str(round(temp/10)*10)
    return dataset


def clean_glucose(dataset, start, stop):
    for i in range(start,stop):
        if dataset.Glucose[i] is None:
            break
        temp=int(dataset.Glucose[i])
        dataset.Glucose[i]=str(round(temp/10)*10)
    return dataset


def clean_bp(dataset, start, stop):
    for i in range(start, stop):
        if dataset.BloodPressure[i] is None:
            break
        temp=int(dataset.BloodPressure[i])
        dataset.BloodPressure[i]=str(round(temp/10)*10)
    return dataset


def clean_dpf(dataset, start, stop):
    for i in range(start, stop):
        if dataset.DiabetesPedigreeFunction[i] is None:
            break
        temp=float(dataset.DiabetesPedigreeFunction[i])
        dataset.DiabetesPedigreeFunction[i]=str(round(temp,2))
    return dataset


def clean_bmi(dataset, start, stop):
    for i in range(start, stop):
        if dataset.BMI[i] is None:
            break
        temp=float(dataset.BMI[i])
        dataset.BMI[i]=str(round(temp))
    return dataset


def clean_ins(dataset, start, stop):
    for i in range(start, stop):
        if dataset.Insulin[i] is None:
            break
        temp=int(dataset.Insulin[i])
        dataset.Insulin[i]=str(round(temp/10)*10)
    return dataset


cleaned = clean_glucose(training_data, 1, 700)
cleaned = clean_age(cleaned, 1, 700)
cleaned = clean_bp(cleaned, 1, 700)
cleaned = clean_dpf(cleaned,1, 700)
cleaned = clean_bmi(cleaned,1, 700)
cleaned = clean_ins(cleaned, 1, 700)

training_data = cleaned  # Training data now becomes the cleaned version of itself
cleaned = clean_glucose(testing_data, 0, 69)
cleaned = clean_age(cleaned, 0, 69)
cleaned = clean_bp(cleaned, 0, 69)
cleaned = clean_dpf(cleaned, 0, 69)
cleaned = clean_bmi(cleaned, 0, 69)
cleaned = clean_ins(cleaned, 0, 69)

testing_data = cleaned  # Testing data now becomes the cleaned version of itself


def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i : abs(lst[i] - K))]


#################################################################################

#################################################################################

'''
    test function helps to find the prediction accuracy of the decision tree by converting the testing_data to a list of dictionary and then finding
    its probability thus and comparing it with the actual output of the dataset. 
'''


def test(data, tree):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = data.iloc[:, :-1].to_dict(orient="records")
    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data["Outcome"]) / len(data)) * 100, '%')


################################################################

################################################################


"""
Train the tree, Print the tree and predict the accuracy
"""


tree = ID3(training_data, training_data, training_data.columns[:-1])
pprint(tree)
test(testing_data, tree)

################################################################

################################################################


'''
    Driver part of the program which executes the above functions and asks the user for their input of data and provide the result whether the person
    has diabetes or not
'''


user_input = {}
root = tk.Tk()
root.geometry("711x621")
root.resizable(False, False)
frame1 = ttk.Frame(root, height=50)
color = "#4EDFE7"
greet = tk.Label(frame1, text="  Diabetes Predictor", compound="left", bg=color, padx=20,
                 pady=10)
greet.config(font=("Garamond", 25))
greet.pack(fill="both", expand=True)
frame1.pack(side="top", fill="x")
color2 = "#323BCA"
frame2 = tk.Frame(root, height=400)
s = ttk.Style()
s.configure('my.TButton', font=('Didot', 12))


def assign():
    global user_input
    global tree
    user_input['Glucose'] = str(int(glu.get()))
    user_input['BloodPressure'] = str(int(bp.get()))
    user_input['SkinThickness'] = str(int(st.get()))
    user_input['Insulin'] = str(int(ins.get()))
    user_input['BMI'] = str(int(bmi.get()))
    user_input['DiabetesPedigreeFunction'] = str(float(dpf.get()))
    user_input['Age'] = str(int(age.get()))
    result = predict(user_input, tree)
    if result == '1':
        sub_root = tk.Toplevel(root)
        sub_root.title("Result")
        ttk.Label(sub_root, text = "Your symptoms suggest possibility of Diabetes", padding = 10).pack()
    else:
        sub_root = tk.Toplevel(root)
        sub_root.title("Result")
        ttk.Label(sub_root, text="Your symptoms do not suggest a strong chance of Diabetes", padding=10).pack()


glu = tk.DoubleVar()
bp = tk.DoubleVar()
st = tk.DoubleVar()
ins = tk.DoubleVar()
bmi = tk.DoubleVar()
dpf = tk.DoubleVar()
age = tk.DoubleVar()
g = ttk.Label(frame2, text="Glucose count :              ", padding=10).grid(row=0, column=0, sticky="we")
b = ttk.Label(frame2, text="Blood Pressure :             ", padding=10).grid(row=1, column=0, sticky="we")
s = ttk.Label(frame2, text="Skin Thickness :             ", padding=10).grid(row=2, column=0, sticky="we")
i = ttk.Label(frame2, text="Insulin :                    ", padding=10).grid(row=3, column=0, sticky="we")
bm = ttk.Label(frame2,text="BMI :                        ", padding=10).grid(row=4, column=0, sticky="we")
d = ttk.Label(frame2, text="Diabetes Pedigree Function : ", padding=10).grid(row=5, column=0, sticky="we")
a = ttk.Label(frame2, text="Age :                        ", padding=10).grid(row=6, column=0, sticky="we")
g_e = ttk.Entry(frame2, width=15, textvariable=glu).grid(row=0, column=1, sticky="we")
b_e = ttk.Entry(frame2, width=15, textvariable=bp).grid(row=1, column=1, sticky="we")
s_e = ttk.Entry(frame2, width=15, textvariable=st).grid(row=2, column=1, sticky="we")
i_e = ttk.Entry(frame2, width=15, textvariable=ins).grid(row=3, column=1, sticky="we")
bm_e = ttk.Entry(frame2, width=15, textvariable=bmi).grid(row=4, column=1, sticky="we")
d_e = ttk.Entry(frame2, width=15, textvariable=dpf).grid(row=5, column=1, sticky="we")
a_e = ttk.Entry(frame2, width=15, textvariable=age).grid(row=6, column=1, sticky="we")
button1 = ttk.Button(frame2, text="Add", command=assign)
button2 = ttk.Button(frame2, text="Cancel", command=root.destroy)
button1.grid(row=7, column=0)
button2.grid(row=7, column=1)
frame2.pack()
root.mainloop()