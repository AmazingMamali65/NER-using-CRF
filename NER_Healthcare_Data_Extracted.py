# Setting up Google Colab for usage. Please disable if running locally.
import pathlib
import os
from google.colab import drive
drive.mount('/content/gdrive')
base_dir = pathlib.Path('/content/gdrive/My Drive/Colab Notebooks/NLP/06_Custom_NER_Medical_Data')
os.chdir(str(base_dir))

!ls

# Installing and importing relevant libraries
!pip install pycrf
!pip install sklearn-crfsuite

import spacy
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pandas as pd

model = spacy.load("en_core_web_sm")

# Reading the train and test sentences and labels
with open('train_sent', 'r') as train_sent_file:
  train_words = train_sent_file.readlines()

with open('train_label', 'r') as train_labels_file:
  train_labels_by_word = train_labels_file.readlines()

with open('test_sent', 'r') as test_sent_file:
  test_words = test_sent_file.readlines()

with open('test_label', 'r') as test_labels_file:
  test_labels_by_word = test_labels_file.readlines()

# Sanity check to see that the number of tokens and no. of corresponding labels match.
print("Count of tokens in training set\n","No. of words: ",len(train_words),"\nNo. of labels: ",len(train_labels_by_word))
print("\n\nCount of tokens in test set\n","No. of words: ",len(test_words),"\nNo. of labels: ",len(test_labels_by_word))

# Function to combine tokens belonging to the same sentence. Sentences are separated by "\n" in the dataset.
def convert_to_sentences(dataset):
    sent_list = []
    sent = ""
    for entity in dataset:
        if entity != '\n':
            sent = sent + entity[:-1] + " "       # Adding word/label to current sentence / sequence of labels 
        else: 
            sent_list.append(sent[:-1])           # Getting rid of the space added after the last entity.
            sent = ""
    return sent_list

# Converting tokens to sentences and individual labels to sequences of corresponding labels.
train_sentences = convert_to_sentences(train_words)
train_labels = convert_to_sentences(train_labels_by_word)
test_sentences = convert_to_sentences(test_words)
test_labels = convert_to_sentences(test_labels_by_word)

print("First five training sentences and their labels:\n")
for i in range(5):
    print(train_sentences[i],"\n",train_labels[i],"\n")

print("First five test sentences and their labels:\n")
for i in range(5):
    print(test_sentences[i],"\n",test_labels[i],"\n")

print("Number of sentences in the train dataset: {}".format(len(train_sentences)))
print("Number of sentences in the test dataset: {}".format(len(test_sentences)))

print("Number of lines of labels in the train dataset: {}".format(len(train_labels)))
print("Number of lines of labels in the test dataset: {}".format(len(test_labels)))

# Creating a combined dataset from training and test sentences, since this is an Exploratory analysis.
combined = train_sentences + test_sentences
print("Number of sentences in combined dataset (training + test): {}".format(len(combined)))

# Creating a list of tokens which have PoS tag of 'NOUN' or 'PROPN'
noun_propn = []         # Initiating list for nouns and proper nouns
pos_tag = []            # initiating list for corresponding PoS tags.
for sent in combined:
    for token in model(sent):
        if token.pos_ in ['NOUN', 'PROPN']:
           noun_propn.append(token.text)
           pos_tag.append(token.pos_)
print("No. of tokens in combined dataset with PoS tag of 'NOUN' or 'PROPN': {}".format(len(noun_propn)))

print(len(pos_tag))

noun_pos = pd.DataFrame({"NOUN_PROPN":noun_propn,"POS_tag":pos_tag})
print("Top 25 comon tokens with PoS tag of 'NOUN' or 'PROPN' \n")
print(noun_pos["NOUN_PROPN"].value_counts().head(25))

# Analysis of PoS tags - Independent assignment for words vs Contextual assignment in a sentence.
sentence = train_sentences[1]   
sent_list = sentence.split()      # Splitting the sentence into its constituent words.
position = 2                      # Choosing position of word within sentence. Index starts at 0.

word = sent_list[position]        # Extracting word for PoS tag analysis.

print(sentence)

# Independent assignment of PoS tag (No contextual info)
print("\nPoS tag of word in isolation\nWord:",word,"--",model(word)[0].pos_,"\n")

# Contextual assignment of PoS tag based on other words in the sentence.
print("PoS tag of all words in sentence with context in tact.")
for token in model(sentence):
    print(token.text, "--", token.pos_)

# Modified workflow to obtain PoS tag of specific word in question while keeping sentence context in tact.
print("\nResult of modified workflow to obtain PoS tag of word at a specific position while keeping context within sentence in-tact.")
cnt = 0                           # Count of the word position within sentence.
for token in model(sentence):
      postag = token.pos_
      if (token.text == word) and (cnt == position):
          break
      cnt += 1
print("Word:", word,"POSTAG:",postag)

# Function to obtain contextual PoS tagger.
def contextual_pos_tagger(sent_list,position):
    '''Obtaining PoS tag for individual word with sentence context in-tact. 
       If the PoS tag is obtained for a word individually, it may not capture the context of use in the sentence and may assign the incorrect PoS tag.'''

    sentence = " ".join(sent_list)          # Sentence needs to be in string format to process it with spacy model. List of words won't work.
    posit = 0                               # Initialising variable to record position of word in joined sentence to compare with the position of the word under considertion.
    for token in model(sentence):
        postag = token.pos_
        if (token.text == word) and (posit == position):
            break
        posit += 1
    return postag

# Define the features to get the feature values for one word.
def getFeaturesForOneWord(sent_list, position):
  word = sent_list[position]
    
  # Obtaining features for current word
  features = [
    'word.lower=' + word.lower(),                                   # serves as word id
    'word.postag=' + contextual_pos_tagger(sent_list, position),    # PoS tag of current word
    'word[-3:]=' + word[-3:],                                       # last three characters
    'word[-2:]=' + word[-2:],                                       # last two characters
    'word.isupper=%s' % word.isupper(),                             # is the word in all uppercase
    'word.isdigit=%s' % word.isdigit(),                             # is the word a number
    'words.startsWithCapital=%s' % word[0].isupper()                # is the word starting with a capital letter
  ]
 
  if(position > 0):
    prev_word = sent_list[position-1]
    features.extend([
    'prev_word.lower=' + prev_word.lower(),                               # previous word
    'prev_word.postag=' + contextual_pos_tagger(sent_list, position - 1), # PoS tag of previous word
    'prev_word.isupper=%s' % prev_word.isupper(),                         # is the previous word in all uppercase
    'prev_word.isdigit=%s' % prev_word.isdigit(),                         # is the previous word a number
    'prev_words.startsWithCapital=%s' % prev_word[0].isupper()            # is the previous word starting with a capital letter
  ])
  else:
    features.append('BEG')                                                # feature to track begin of sentence 
 
  if(position == len(sent_list)-1):
    features.append('END')                                                # feature to track end of sentence
 
  return features

# Write a code to get features for a sentence.
def getFeaturesForOneSentence(sentence):
  sentence_list = sentence.split()
  return [getFeaturesForOneWord(sentence_list, position) for position in range(len(sentence_list))]

# Checking feature extraction
example_sentence = train_sentences[5]
print(example_sentence)

features = getFeaturesForOneSentence(example_sentence)
features[0]

features[4]

# Write a code to get the labels for a sentence.
def getLabelsInListForOneSentence(labels):
  return labels.split()

# Checking label extraction
example_labels = getLabelsInListForOneSentence(train_labels[5])
print(example_labels)

X_train = [getFeaturesForOneSentence(sentence) for sentence in train_sentences]
X_test = [getFeaturesForOneSentence(sentence) for sentence in test_sentences]

Y_train = [getLabelsInListForOneSentence(labels) for labels in train_labels]
Y_test = [getLabelsInListForOneSentence(labels) for labels in test_labels]

# Building the CRF model. Using max_iterations as 200.
crf = sklearn_crfsuite.CRF(max_iterations=300)

crf.fit(X_train, Y_train)

Y_pred = crf.predict(X_test)

metrics.flat_f1_score(Y_test, Y_pred, average='weighted')

# Example test sentence and corresponding actual and predicted labels 
print("Sentence: ",test_sentences[13])
print("Actual labels:    ", Y_test[13])
print("Predicted labels: ", Y_pred[13])

# Feature list of sentence above
print(X_test[13])

# Extracting a dictionary of all the predicted diseases from our test data and the corresponding treatments.
# Assumption: For each identified disease, one of the treatments is in the same sentence as the disease exists.
disease_treatment = {}            # Initializing an empty dictionary
for i in range(len(Y_pred)):
    cnt_disease = 0           # Count of number of diseases mentioned in the sentence
    cnt_treatment = 0         # Count of the number of treatments mentioned in the sentence
    diseases = [""]           # Initializing a blank list of diseases for current sentence.
    treatment = [""]          # Initializing a blank list of treatments for current sentence.
    length = len(Y_pred[i])   # Length of current sentence.
    for j in range(length):
        if (Y_pred[i][j] == 'D'):                                                     # Checking for label indicating disease for current word ('D')
            diseases[cnt_disease] += (X_test[i][j][0].split('=')[1] + " ")            # Adding word to diseases list.
            if j < length - 1:
                if (Y_pred[i][j+1] != 'D'):                                           # Check for name of disease extending over multiple words. 
                    # If next word does not have label 'D', then truncate the space added at the end of the last word.
                    diseases[cnt_disease] = diseases[cnt_disease][:-1]
                    cnt_disease += 1
                    diseases.append("")                                               # Adding a placeholder for the next disease in the current sentence.
            else:
                diseases[cnt_disease] = diseases[cnt_disease][:-1]
                cnt_disease += 1
                diseases.append("")
                            
        if (Y_pred[i][j] == 'T'):                                                     # Checking for label indicating treatment for current word ('T')
            treatment[cnt_treatment] += (X_test[i][j][0].split('=')[1] + " ") # Adding word to corresponding treatment list.
            if j < length - 1:
                if (Y_pred[i][j+1] != 'T'):                                           # Check for name of treatment extending over multiple words. 
                    # If next word does not have label 'T', then truncate the space added at the end of the last word.
                    treatment[cnt_treatment] = treatment[cnt_treatment][:-1]
                    cnt_treatment += 1
                    treatment.append("")                                              # Adding a placeholder for the next treatment in the current sentence.
            else:
                treatment[cnt_treatment] = treatment[cnt_treatment][:-1]
                cnt_treatment += 1
                treatment.append("")

    diseases.pop(-1)    # Getting rid of the last empty placeholder in diseases list
    treatment.pop(-1)   # Getting rid of the last empty placeholder in treatments list

    # To our dictionary, add or append treatments to the diseases identified from the current sentence, if any.
    if len(diseases) > 0:       # Checking if any diseases have been identified for the current sentence.
        for disease in diseases:
            if disease in disease_treatment.keys():
                # Extend treatment list if other treatments for the particular disease already exist
                disease_treatment[disease].extend(treatment)
            else:
                # Creating list of treatments for particular disease if it doesn not exist already.
                disease_treatment[disease] = treatment

# Displaying dictionary of extracted diseases and potential treatments.
disease_treatment

# Obtaining a cleaned version of our "disease_treatment" dictionary
cleaned_dict = {}
for key in disease_treatment.keys():
    if disease_treatment[key] != []:
        cleaned_dict[key] = disease_treatment[key]
cleaned_dict

# Converting dictionary to a dataframe
cleaned_df = pd.DataFrame({"Disease":cleaned_dict.keys(),"Treatments":cleaned_dict.values()})
cleaned_df.head()

search_item = 'hereditary retinoblastoma'
treatments = cleaned_dict[search_item]
print("Treatments for '{0}' is/are ".format(search_item), end = "")
for i in range(len(treatments)-1):
    print("'{}'".format(treatments[i]),",", end="")
print("'{}'".format(treatments[-1]))