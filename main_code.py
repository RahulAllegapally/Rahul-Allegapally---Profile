from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle
import pandas as pd
import numpy as np
path_training_data = os.getcwd()+'/training_data.tsv'
path_test_data=os.getcwd()+'/eval_data.txt'
###### regex generation ######
import regex_generator

regex_generator.main_regex_generator(path_training_data)

####### regex extraction ######
import regex_matcher

print('inside_extration....This might take a while')
extractions_from_regex=regex_matcher.main_regex_matcher(path_test_data)
print('extracted')
###### training the ML model ######
import training_classifier
###### get trained classifier model ######
training_classifier.main_ML_model()
f = open('my_classifier.pickle', 'rb')
clf = pickle.load(f)
f.close()
###### get vectorizer ######
f = open('my_vectorizer.pickle', 'rb')
vectorizer = pickle.load(f)
f.close()

###### loading data ######
data=[]
master_data=open(path_test_data,'r')
for sent in master_data:
    data.append(' '.join(sent.split()))#using ' '.join(sent.split()) to remove the \n from the txt file
df=pd.DataFrame(data) # loading data to the dataframe
df=df[0]# getting the dimentions right for prediction
#
#
#
clean_test_data = []
for i in df:
    clean_test_data.append(i)
# print (clean_train_data)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_data)
np.asarray(test_data_features)

tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(test_data_features)

###### predicting from the classifier ######

result = clf.predict(X_test_tfidf)

###### saving the result in a dataframe ######

output = pd.DataFrame(data={"sent": df, "label": result})
predicted_result=list(result) # converting the n dimentional array into a list
###### output of the classisfier using BOW model ######
output.to_csv(('Bag_of_Words_model_new.csv'), index=False, quoting=3, escapechar='\\')
###### final submission ######
final_result=[]
for idx,prediction in enumerate(predicted_result):
    if (prediction=='Not Found'):
        final_result.append('Not Found') # if classifier identifies the sentence to be labled as Not Found it has power to over write the extracter
    else:
        final_result.append(extractions_from_regex[idx])# if the classifier identifies it to have a phrase then we use the extracted phrase
###### loading the final output into df ######
final_output = pd.DataFrame(data={"sent": df, "label": final_result})
###### Final submission ######
final_output.to_csv(('submission.csv'), index=False, quoting=3, escapechar='\\')
print("submission done")