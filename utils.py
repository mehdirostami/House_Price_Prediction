
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform

import os
from sklearn.metrics import roc_curve, auc
from numpy.random import normal, multivariate_normal
from cvxopt import matrix, solvers
import warnings
from sklearn.cluster import AgglomerativeClustering
import multiprocessing 
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, make_scorer
import warnings
warnings.filterwarnings("ignore")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# from nn_bnn_causality import NNLinear, BNN, NN, justBNN, BNNx

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from glmnet import ElasticNet, LogitNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from mlens.ensemble import SuperLearner, Subsemble
from mlens.model_selection import Evaluator

import tabulate
# tabulate.LATEX_ESC
import re
APE_RULES={}



def to_latex_table(df, num_header=2, remove_char=[]):

    if num_header == 2:
        df2 = df.reset_index().T.reset_index().T
    else:
        df2 = df.T.reset_index().T
    remove_cols = ['0', 'index']
    # print(df2)
    # for col in remove_cols:
    #     if col in list(df2):
    #         df2 = df2.drop(col, axis=1)
    #         print(col)

    out = tabulate.tabulate(df2.reset_index(), tablefmt='latex_raw')
    out = out.replace("\\\\", "\\")
    out = out.replace("\\\\", "\\")
    out = re.sub(' +', ' ', out)

    for char in remove_char:
        out = out.replace(char, "")

    # out = eval(out)
    return(out)



def brier(obs_binary, pred_prob):
    return(((pred_prob - obs_binary)**2).mean())


def psudor2(obs_binary, pred_prob):
    return(np.abs((obs_binary * pred_prob).mean() - ((1 - obs_binary) * pred_prob).mean()))











# import torchsample

def plot(param_dict, layer_names, num_plots):
	"""
	layer_names is the title of weights in the state_dict() for different layers.
	num_plots is the number of histograms for each layer
	"""

	params = param_dict

	#since params is a dictionary of tensors, to get the size of each tensor saved in it, we'll use .size()
	for layer in layer_names:
		if params[layer].size(0) < num_plots:
			raise AssertionError #"Number of plots for a layer should be less than or equal to the size of that layer."
	
	
	fig, multi_plots = plt.subplots(nrows=len(layer_names), ncols=num_plots, sharex=True)

	for i in range(len(layer_names)):
		for j in range(num_plots):
			multi_plots[i, j].hist(params[layer_names[i]][j, :])


	if not os.path.exists("saved_plots"):
		os.makedirs("saved_plots")
	plt.savefig('./saved_plots/mlp_mnist.png')
	plt.show()
	

def get_batch(x, y, batch_size):
    '''
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    '''
    N = x.shape[0]
    assert N == y.shape[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i+batch_size, :]
        batch_y = y[i:i+batch_size]
        yield (batch_x, batch_y)


def plot_learning_curve(train_loss, valid_loss):
	
	e = train_loss.shape[0]
	plt.plot(range(e), train_loss)
	plt.plot(range(e), valid_loss)
	plt.show()

	

def binary_accuracy(y, t, threshold = .5):
	"""y and t are tensors"""
	y_cat = 0 + (y >= threshold)
	#a11 = torch.sum(y_cat * t); a12 = torch.sum(y_cat * (1 - t))
	#a21 = torch.sum((1 - y_cat) * t); a22 = torch.sum((1 - y_cat) * (1 - t))
	a11 = torch.dot(y_cat, t); a12 = torch.dot(y_cat, (1 - t))
	a21 = torch.dot((1 - y_cat), t); a22 = torch.dot((1 - y_cat), (1 - t))
	print("Confusion matrix (predicted vs. observed):")
	confuse = torch.Tensor([[a11, a12], [a21, a22]])
	print(confuse)
	print("Sensitivity (%):", np.round(100*a11/(a11 + a21), 1))
	print("Specificity (%):", np.round(100*a22/(a22 + a12), 1))
	print("Accuracy (%):", np.round(100*(a11 + a22)/torch.sum(confuse), 1))

#def accuracy(y, t, threshold = .5):
#	"""y and t are tensors"""
#	y = y.data.numpy()
#	t = t.data.numpy()
#	y_cat = 0. + (y >= threshold)
#	a11 = np.dot(y_cat.T, t);
#	a12 = np.dot(y_cat.T, (1. - t))
#	a21 = np.dot((1. - y_cat).T, t)
#	a22 = np.dot((1. - y_cat).T, (1. - t))
#	print("Confusion matrix (predicted vs. observed):")
#	#confuse = np.array([[a11, a12], [a21, a22]])
#	confuse = np.vstack((np.hstack((a11, a12)), np.hstack((a21, a22))))
#	print(confuse)
#	print("Sensitivity (%):", np.round(100*a11/(a11 + a21), 1))
#	print("Specificity (%):", np.round(100*a22/(a22 + a12), 1))
#	print("Accuracy (%):", np.round(100*(a11 + a22)/np.sum(confuse), 1))
#	return("------------------------------------------")

def ROC_AUC(predprob, observed):
	fpr, tpr, _ = roc_curve(observed, predprob)
	auc = auc(fpr, tpr)
	return(fpr, tpr, auc)


###############################################################################
###############################################################################
# Simulating data
###############################################################################
###############################################################################

class EnsembleCV(object):
    """docstring for EnsembleCV"""
    def __init__(self, X, y, ML_methods, score_method, fold=3, seed=123, print_warnings=False):
        super(EnsembleCV, self).__init__()
        if not print_warnings:
            warnings.filterwarnings("ignore")
        
        self.X = X
        self.y = y
        self.ML_methods = ML_methods
        self.score_method = score_method
        self.fold = fold
        self.seed = seed
        self.n, self.p = self.X.shape

    def fit(self, X_train, y_train):
        objects = []
        
        for name, ML in self.ML_methods:
            objects += [(name, ML.fit(X_train, y_train))]
        
        return(objects)

    def cv(self):

        ML_methods_ = []
        self.scores = {}
        self.y_pred = {}
        
        if isinstance(self.ML_methods, dict):
            for key, ML in self.ML_methods.items():
                ML_methods_ += [(key, ML)]
                self.y_pred[key] = np.ones((self.n, 1))
        else:
            for i, ML in enumerate(self.ML_methods):
                ML_methods_ += [(str(i), ML)]
                self.y_pred[str(i)] = np.ones((self.n, 1))       
        
        self.ML_methods = ML_methods_

        np.random.seed(self.seed)
        self.splits = ((self.fold * np.random.rand(self.n, 1)).astype(int) % self.fold).flatten()
        
        for k in np.unique(self.splits):
            
            self.scores["fold "+str(k)] = {}
            
            train_index = np.where(self.splits != k)[0]
            X_train, y_train = self.X[train_index, :], self.y[train_index].reshape(-1, 1)
            
            test_index = np.where(self.splits == k)[0]
            X_test, y_test = self.X[test_index, :], self.y[test_index].reshape(-1, 1)
            
            objects = self.fit(X_train, y_train)
            
            for obj_name, obj in objects:
                self.y_pred[obj_name][test_index] = obj.predict(X_test).reshape(-1, 1)
                self.scores["fold "+str(k)][obj_name] = self.score_method(self.y_pred[obj_name][test_index].reshape(-1, 1), self.y[test_index].reshape(-1, 1))
        
        col_names = [key for key, _ in self.scores.items()]
        row_names0 = [key for _, value in self.scores.items() for key, _ in value.items()]
        num_uniq_algs = len(set(row_names0))
        row_names = row_names0[:num_uniq_algs]
        
        self.final_scores = pd.DataFrame([[v for _,v in value.items()] for _,value in self.scores.items()]).T
        
        self.final_scores.columns = col_names
        self.final_scores.index = row_names
        
        overall_scores = []
        for obj_name, _ in objects:
            overall_scores += [self.score_method(self.y_pred[obj_name], self.y.reshape(-1, 1))]
        
        self.final_scores['Overall'] = np.array(overall_scores).reshape(-1, 1)
        
        return(self)

    def SuperLearner(self, inputs=False):
    
        all_preds = np.concatenate([cols for names, cols in self.y_pred.items()], axis=1)
        print([names for names, cols in self.y_pred.items()])
        
        if inputs is False and type(inputs) == bool:
            matrix_preds = all_preds
        elif inputs is True and type(inputs) == bool:
            matrix_preds = np.concatenate([all_preds, self.X], axis=1)
        else:
            matrix_preds = np.concatenate([all_preds, inputs], axis=1)
        
        supl = Lasso(alpha=.0001, positive=True, fit_intercept=False)
        supl.fit(matrix_preds, self.y)
        
        self.supl_pred = supl.predict(matrix_preds) 
        
        coefficients = supl.coef_
        self.coef_ = coefficients/coefficients.sum()
        
        self.supl_score = self.score_method(self.supl_pred, y)
        
        return(self)


class CleanIt(object):
    """
    A pipeline for automatic preprocessing of a given dataset.

    data = pd.read_csv("./train.csv", sep=',')

    preprocess = utils.CleanIt(df=data, outcome='saleprice')

    preprocess.yyyy_mm_toDays(yyyy='yrsold', mm='mosold', output_time_var='time_to_sold', origin='20000101')

    preprocess.LowercaseStringVars()

    preprocess.CollapseCategories(fill_nan_value=9999, collapse_ratio=.05)

    preprocess.DummyIt(include_old_cols=False)

    train_x, test_x, train_y, test_y = preprocess.SplitIt(vars_to_omit=['id'],standardizeIt=True, train_size=.8, seed=123)






    """
    def __init__(self, df, outcome=''):
        super(CleanIt, self).__init__()

        self.df = df.copy()
        self.outcome = outcome


        self.df.columns = [col.lower() for col in self.df.columns]
        self.categorical_vars = list(self.df.describe(include=['object', 'bool', 'category']).columns)
        self.continuous_vars = [col for col in self.df if col not in self.categorical_vars]

        self.collased_keyword = '_collapsed_'

    def timedeltaToDay(self, col):
        return(col.dt.total_seconds().astype(int)/(60*60*24))
        
    def yyyy_mm_toDays(self, yyyy, mm, output_time_var, df=None, origin='20000101'):

        if df is None:
            temp_df = self.df.copy()
        else:
            temp_df = df.copy()

        temp_df['yyyymmdd0'] = temp_df[yyyy]*10000 + temp_df[mm]*100 + 15 # Middle of month
        temp_df['yyyymmdd'] = pd.to_datetime(temp_df['yyyymmdd0'], format='%Y%m%d')
        temp_df[output_time_var] = self.timedeltaToDay(temp_df['yyyymmdd'] - pd.to_datetime(origin, format='%Y%m%d'))
        # temp_df = temp_df.drop([yyyy, mm, 'yyyymmdd0', 'yyyymmdd'], axis=1)
        # temp_df = temp_df.drop([yyyy, 'yyyymmdd0', 'yyyymmdd'], axis=1)
        # temp_df = temp_df.drop(['yyyymmdd0', 'yyyymmdd'], axis=1)
        temp_df = temp_df.drop([mm, 'yyyymmdd0', 'yyyymmdd'], axis=1)

        if df is None:
            self.df = temp_df.copy()
            return(self)
        else:
            return(temp_df)


    def DescribeCategorical(self, df=None):

        if df is None:
            return(self.df.describe(include=['object', 'bool', 'category']))
        else:
            print('-----')
            print(df.shape)
            return(df.describe(include=['object', 'bool', 'category']))
            print('-----')


    def LowercaseStringVars(self, df=None):
        
        if df is None:
            df_lowercase = self.df.copy()
        else:
            df_lowercase = df.copy()

        for var in self.categorical_vars:
            df_lowercase[var] = df_lowercase[var].astype(str).str.lower()

        if df is None:
            self.df = df_lowercase.copy()
            return(self)
        else:
            return(df_lowercase)


    def FillIt(self, df, fill_nan_value=False):

        if fill_nan_value is not False:
            df = df.fillna(fill_nan_value)
        elif fill_nan_value is False:
            df = df.fillna(0.)

        return(df)


    def CollapseCategories(self, fill_nan_value=9999, collapse_ratio=.0, collapse_categories_dict={}, df=None):#https://stackoverflow.com/questions/47418299/python-combining-low-frequency-factors-category-counts
        
        if len(collapse_categories_dict) > 0 and collapse_ratio > 0.:
            raise ValueError("only one of the methods can be used: collapsing based on proportion or on given categories in the dictionary (collapse_categories_dict)")
        df_collapsed = self.df.copy()

        df_collapsed = self.FillIt(df_collapsed)

        if len(collapse_categories_dict) > 0:
            for column, values in collapse_categories_dict.items():
                for tuple_ in values:
                    df_collapsed[column] = df_collapsed[column].astype(str)
                    df_collapsed[column] = df_collapsed[column].replace(tuple_[1], tuple_[0])
        
        if collapse_ratio > 0.:
            df_collapsed[self.categorical_vars] = df_collapsed[self.categorical_vars].astype(str)
            for col in self.categorical_vars:
                
                temp = pd.value_counts(df_collapsed[col])
                temp_ratio = (temp/temp.sum()).lt(collapse_ratio)
                
                df_collapsed[col] = np.where(df_collapsed[col].isin(temp[temp_ratio].index), self.collased_keyword, df_collapsed[col])

        self.df = df_collapsed.copy()
        self.train_df_collapsed_notDummy = df_collapsed.copy()

        return(self) 


    def CollapseTestData(self, df):

        for col in self.categorical_vars:
            # print(col)
            # print('---------------------')
            # print(df[col].unique())
            # print('---------------------')
            # print(self.train_df_collapsed_notDummy[col].unique())
            # print('---------------------')
            temp_list = [str(val) for val in df[col].unique() if val not in self.train_df_collapsed_notDummy[col].unique()]
            if len(temp_list) >= 1:
                df[col] = df[col].replace(regex=temp_list, value=self.collased_keyword)

        return(df)

    def Nominal2Ordinal(self, df=None):

        if df is None:
            df1 = self.df.copy()
        else:
            df1 = df

        self.numeric_categorical_vars = list(df1.describe(include=['object', 'bool', 'category']).columns)
        
        for col in self.numeric_categorical_vars:
            temp_encoder = LabelEncoder()
            df1[col] = temp_encoder.fit_transform(df1[col])

        if df is None:
            self.df = df1.copy()
            return(self)
        else:
            return(df1)

    def DummyIt(self, df=None, tobinarize=[], include_old_cols=False, DropLowVarDummyThreshold=0.):
        """Create a table with old columns and new columns that are binary variables of old categorical variables.
        categorical variables should be string"""
        if df is None:
            df1 = self.df.copy()
        else:
            df1 = df

        dumm_str = [pd.get_dummies(df1[col], prefix='i_'+col) for col in tobinarize]

        df_dummies = pd.concat(dumm_str, axis=1)

        for col in df_dummies.columns:
            if df_dummies[col].var() < DropLowVarDummyThreshold:
                df_dummies = df_dummies.drop(col, axis=1)

        if include_old_cols:
            out = pd.concat([df1, df_dummies], axis=1)

        else:
            not_binarized = [cols for cols in df1.columns if cols not in tobinarize]
            out = pd.concat([df1[not_binarized], df_dummies], axis=1)

        if df is None:
            self.df = out.copy()
            return(self)
        else:
            return(out)

    def AddSummations(self, df=None, vars_dict={}, keep_old_vars=False, prefix='sum_'):

        if df is None:
            df1 = self.df.copy()
        else:
            df1 = df

        for key, values in vars_dict.items():
            df1[prefix+key] = df1[values].sum(1)
            if keep_old_vars:
                df1 = df1.drop(values, axis=1)

        if df is None:
            self.df = df1.copy()
            return(self)
        else:
            return(df1)

    def CategorizeIt(self, df=None, categories_dict={}, prefix='cat_'):

        if df is None:
            df1 = self.df.copy()
        else:
            df1 = df

        for col, cutoffs in categories_dict.items():
            df1 = CategorizeIt(df1, col, cutoffs, first_value=0, last_value="", prefix=prefix)

        if df is None:
            self.df = df1.copy()
            return(self)
        else:
            return(df1)


    def AddInteractions(self, df=None, degree=3, interaction_only=False, include_bias=False, exclude_cols=[], DropLowVarInteractionsThreshold=.0):

        if df is None:
            df1 = self.df.copy()
        else:
            df1 = df.copy()

        temp_df1 = df1[[col for col in df1.columns if col not in exclude_cols]].copy()

        poly_features = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        all_main_nonlin = poly_features.fit_transform(temp_df1)

        all_main_nonlin = pd.DataFrame(all_main_nonlin)
        all_main_nonlin.columns = [cols for cols in temp_df1] + [str(i) for i in range(all_main_nonlin.shape[1] - temp_df1.shape[1])]
        all_main_nonlin.index = df1.index

        cols_to_delete = [col for col in all_main_nonlin.columns if col.isdigit() and all_main_nonlin[col].var() < DropLowVarInteractionsThreshold]
        if len(cols_to_delete) > 0 and DropLowVarInteractionsThreshold > 0.:
            all_main_nonlin = all_main_nonlin.drop(cols_to_delete, axis=1)

        all_main_nonlin = all_main_nonlin.T.drop_duplicates().T
        
        if df is None:
            self.df = pd.concat([all_main_nonlin, df1[exclude_cols]], axis=1).copy()
            return(self)
        else:
            return(pd.concat([all_main_nonlin, df1[exclude_cols]], axis=1).copy())


        

    def DropLowVar(self, df=None, threshold=.01):

        if df is None:
            df1 = self.df.copy()
        else:
            df1 = df.copy()

        toBeDroped = [col for col in df1.columns if df1[col].var() < threshold]
        
        df1 = df1.drop(toBeDroped, axis=1)

        if df is None:
            self.df = df1.copy()
            return(self)
        else:
            return(df1)



def ScaleTrainData(train_data):
    scaleIt = StandardScaler()
    train_x = scaleIt.fit_transform(train_data)
    return(scaleIt, train_x)

def ScaleTestData(scaleIt, test_x_data):
    return(scaleIt.transform(test_x_data))

def SplitIt(df_x, y, exclude_cols=['id'], standardizeIt=True, train_size=.8, seed=123):

    X = df_x[[col for col in df_x.columns if col not in exclude_cols]]

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=seed, train_size=train_size)
    if standardizeIt:
        scaleIt, train_x = ScaleTrainData(train_x)
        test_x = ScaleTestData(scaleIt, test_x)

    return(train_x, test_x, train_y, test_y, scaleIt)


from glmnet import ElasticNet, LogitNet
from pygam.pygam import LinearGAM 

def FeatureSelection(df_x, xtrain, ytrain, exclude_cols=[]):
    
    # #### Gam
    # gam = LinearGAM(n_splines=4).gridsearch(xtrain, ytrain)
    # pvalues = np.array(gam.statistics_['p_values'])
    # important_x_idx_gam = [idx-1 for idx in np.where(pvalues < 0.1)[0]]
    # important_x_gam = df_x.iloc[:, important_x_idx_gam]

    #### Lasso
    lasso = ElasticNet(alpha=1, n_splits=10, random_state=123, n_jobs=4)
    lasso.fit(xtrain, ytrain)
    coeffs = lasso.coef_
    important_x_dx_lasso = np.where(coeffs != 0.)[0]
    important_x_lasso = [col for col in df_x.columns[important_x_dx_lasso]]
    return(important_x_lasso, important_x_dx_lasso)
    # important_features = list(set([col for col in important_x_lasso.columns] + [col for col in important_x_gam.columns]))
    # important_features_idx = list(set([idx for idx in important_x_dx_lasso] + [idx for idx in important_x_idx_gam]))
    # return(important_features, important_features_idx)

def ifelse(conditions, values):
    
    if not isinstance(conditions, list) or not isinstance(conditions, list):
            raise ValueError("The conditions and values should be lists of boolian pandas columns and numbers, respectively.")
    
    if len(conditions) != len(values) - 1:
        raise ValueError("The number of conditions should be number of values minus 1.")
        
    if len(conditions) == 1:
        return(np.where(conditions[0], values[0], values[1]))
    else:
        return(np.where(conditions[0], values[0], ifelse(conditions[1:], values[1:])))


def CategorizeIt(df, varname, cutoffs, first_value=0, last_value="", prefix='cat_'):
    """
    Inputs:
        variable: The variable or variables to be categorized (shape=n*p).
        cutoffs: The list of cutoff values.
    Output: a pandas dataframe with two columns, old variable (copy or input variable) and new variable, which is 
        the categorized version of variable (shape=n*2p). 
    """
    
    cutoffs = np.sort(np.array(cutoffs))
    
    if not isinstance(varname, str):
        raise ValueError("varname should be a string representing the name of the column in df.")             
                
    updated_cutoffs = cutoffs.copy()
    conditions = [(df[varname] <= updated_cutoffs[0])]
    values = [first_value]
    
    j = first_value
    k = len(cutoffs)
    for c in range(k-1):
        conditions = conditions + [(df[varname] > updated_cutoffs[0]) & (df[varname] <= updated_cutoffs[1])]
        values += [j+1]
        j += 1
        updated_cutoffs = updated_cutoffs[1:]

    values += [j+1]
    
    df[prefix + varname] = ifelse(conditions, values)
    
    # banding = BandIt(df=df, varname=varname, cutoffs=cutoffs, first_value=first_value, last_value=last_value)
    
    out = df # pd.merge(df, banding, on="cat_"+varname, how='left')
    new_vars = list(out)
    for var in new_vars:
        out[var] = np.where(out[varname].isnull(), np.nan, out[var])
    
    return(out)   
    
# seed = 12345
# # np.random.seed(seed)
# # X = np.random.normal(size=(1000, 10))
# # beta = np.random.uniform(size=10).reshape(-1, 1)
# # y = np.dot(X, beta) + np.random.normal(size=(1000, 1))

# # pr = 1./(1 + np.exp(-np.dot(X, beta)))
# # y = np.random.binomial(1, pr)


# # ML_methods = {'linreg':LinearRegression(), \
# #               'glmnet': ElasticNet(alpha=1, n_splits=3, n_jobs=3, random_state=123), \
# #               'rf': RandomForestRegressor(n_estimators=10, max_depth=3, max_features='sqrt', random_state=1234),\
# #               'xgb': XGBRegressor(), \
# #               'svm': SVR()
# #              }

# # ensemble = EnsembleCV(X, y, ML_methods=ML_methods, score_method=r2_score, fold=3, seed=123, print_warnings=False)

# # ensemble.cv()
# # print(ensemble.final_scores)


# # ensemble.SuperLearner()
# # print(ensemble.coef_)
# # print(ensemble.supl_score)


# from sklearn.datasets import load_iris

# seed = 2017
# np.random.seed(seed)

# data = load_iris()
# idx = np.random.permutation(150)
# X = data.data[idx]
# y = data.target[idx]

# #### SuperLearner(folds=2, shuffle=False, random_state=None, scorer=None, raise_on_exception=True, array_check=2, verbose=False, n_jobs=-1, backend=None, layers=None)
# #### number of folds to use during fitting. Note: this parameter can be specified on a layer-specific basis in the add method.


# ensemble = SuperLearner(folds=2, scorer=accuracy_score, random_state=seed)

# lev0_logit = [LogisticRegression()]
# lev0_rf = [RandomForestClassifier(n_estimators=nest, max_depth=mx, random_state=1234) for nest in range(10, 500, 100) for mx in range(3, 7, 1)]

# ensemble.add(lev0_logit + lev0_rf )


# ensemble.add(lev0_rf)


# ensemble.add_meta(Lasso(alpha=1))

# ensemble.fit(X, y)

# print("Fit data:\n%r" % ensemble.data)


# exit()
# ensemble = SuperLearner(folds=5, scorer=accuracy_score, random_state=seed)
# ensemble.add([RandomForestClassifier(random_state=seed), LogisticRegression()])
# ensemble.add([LogisticRegression(), SVC()])
# ensemble.add_meta(Lasso(alpha=1))

# ensemble.fit(X[:75], y[:75])

# print("Fit data:\n%r" % ensemble.data)

# preds = ensemble.predict(X[75:])

# exit()

# ###################################################################################################
# ###################################################################################################
# ###################################################################################################


# sub = Subsemble(partitions=3, folds=2)
# sub.add([SVC(), RandomForestClassifier()])
# sub.add([KNeighborsClassifier(), LogisticRegression()])
# sub.add_meta(Lasso(alpha=1))


# sub.fit(X, y)
# print(sub.predict(X))

# exit()


# ###################################################################################################
# ###################################################################################################
# ###################################################################################################


# accuracy_scorer = make_scorer(accuracy_score, greater_is_better=True)


# ests = [('logit', LogisticRegression()), \
#         ('glmnet', LogitNet(alpha=1, n_splits=5, random_state=123)), \
#         ('rf', RandomForestClassifier(max_features='sqrt', random_state=1234)), \
#         ('xgb', XGBClassifier(subsample=.7)), \
#         ('svm', SVC(kernel='poly')), \
#         ('gnb', GaussianNB()), \
#         ('knn', KNeighborsClassifier())
#         ]



# params = {  
#             # 'rf': {'n_estimators': randint(100, 1000), 'max_depth': randint(2, 4)},
#             # 'xgb': {'max_depth': randint(2, 7), 'gamma': uniform(.01, .1), 'eta': uniform(0.01, 0.4), 'lambda': uniform(.5, 1), 'alpha': uniform(.5, 1)},
#             # 'svm': {'C': randint(1, 1000), 'gamma': uniform(0.0001, 0.001), 'degree': randint(2, 5),},
#             # 'knn': {'n_neighbors': randint(2, 20)}
#         }



# evaluator = Evaluator(accuracy_scorer, cv=2, random_state=seed, verbose=1, n_jobs=4)
# evaluator.fit(X, y, ests, params, n_iter=1)


# print("Score comparison with best params founds:\n\n%r" % evaluator.results)

# print(dir(evaluator))


# ###################################################################################################
# ###################################################################################################
# ###################################################################################################