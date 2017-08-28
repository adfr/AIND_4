import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    #  return probabilities, guesses
    #loop over the test set
    for i in range(0,len(test_set.get_all_Xlengths()) ):
        (X, lengths) =  test_set.get_item_Xlengths(i)   

        temp_d= dict()
        #then loop over the models
        max_score = np.float('-inf')
        for word,model in models.items():
           if(max_score==np.float('-inf')):
               max_score= word
           try:
               temp_d[word]=model.score(X, lengths)
               # take the best model
               if temp_d[word] > temp_d[max_score]:
                   max_score = word
           except:
               temp_d[word]= np.float('-inf')
        probabilities.append(temp_d)
        guesses.append(max_score)
    return probabilities, guesses
