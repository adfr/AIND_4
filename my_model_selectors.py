import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic =np.float('inf')
        best_model = None

        for i in range(self.min_n_components, self.max_n_components+1):
            avg_test = 0
            try:
                hmm_model = self.base_model(i)
                logL = hmm_model.score(self.X, self.lengths)
                p = i**2 + 2*i*len((self.lengths)) - 1
                logN = np.log(len((self.lengths)))
                bic = -2 * logL + p * logN
                if bic < min_bic:
                    min_bic = bic
                    best_model = hmm_model  
            except: 
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, i))
                pass
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(self.sequences) == 1:
            return self.base_model(self.n_constant)
        opt_n = 1
        opt_Score = np.float('-inf')
        best_model = self.base_model(self.min_n_components)
        for i in range(self.min_n_components, self.max_n_components+1):
           try:
               hmm_model=self.base_model(i)
               current_logL = hmm_model.score(self.X, self.lengths)
               total_logL=0
               count=0
               for s_word in self.hwords:
                   if s_word != self.this_word:
                       total_logL +=hmm_model.score(self.hwords[s_word])
                       count=+1
               total_logL = total_logL/count
               dic_score=current_logL-total_logL
               if dic_score>opt_Score:
                   opt_Score = dic_score
                   best_model = hmm_model
           except: 
               if self.verbose:
                   print("failure on {} with {} states".format(self.this_word, i))
   
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        Basically we divide between calibration sets and testing sets
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if len(self.sequences) == 1:
            return self.base_model(self.n_constant)
        opt_n = 1
        opt_logScore = np.float('-inf')
        split = min(3, len(self.sequences)) 
        split_method = KFold(split)
        for i in range(self.min_n_components, self.max_n_components+1):
            avg_test = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_set, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                test_set, test_lengths   = combine_sequences(cv_test_idx, self.sequences)
                try:
                    hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(train_set, train_lengths)
                    logScore = hmm_model.score(test_set, test_lengths)
                    avg_test += logScore
                    if self.verbose:
                         print("model created for {} with {} states".format(self.this_word, i))
                except: 
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, i))
            avg_test = avg_test/split
            if avg_test >= opt_logScore:
                opt_logScore = avg_test
                opt_n = i
        best_model = GaussianHMM(n_components=opt_n, covariance_type="diag", n_iter=1000,
                                 random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

        return best_model
