""" 
Methods to describe attriubutes in a dataset. The last column
of a dataset is assumed to be the dependent variable. 

Methods range from but not restricted to:
 - Description of dataset as a whole
 - Description of individual attributes
 - Description of inter-relation between attributes.

Contact: Harsh Nisar GH: harshnisar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Dataset:
    """
    Initialize the dataset and give user the option to set some
    defaults eg. names of categorical columns

    All public methods will provide one value per dataset.
    Private methods are for internal use.  

    prediction_type = {'regression'|'classification'}
    """
    df = None
    df_encoded = None
    categorical_cols = None
    dependent_col = None
    prediction_type = None
    independent_col = None
    def __init__(self, df, prediction_type = None, dependent_col = None,categorical_cols = None):
        
        self.df = df
        self._set_dependent_col(dependent_col)
        self._set_categorical_columns(categorical_cols)
        self._set_prediction_type(prediction_type)

        self.independent_col = list(set(self.df.columns.tolist()) - set([self.dependent_col]))
        self._categorical_column_encoder()

        
    def _set_dependent_col(self, dependent_col):
        """ if nothing given, set the last column in the frame as
        the dependent column."""

        if dependent_col == None:
            self.dependent_col = self.df.columns.tolist()[-1]
        elif dependent_col in self.df.columns.tolist():
            self.dependent_col = dependent_col
        else:
            raise ValueError

    def _set_prediction_type(self, prediction_type):
        """ See the dtype of the dependent_col and return
        either regression or classification 
        """
        if prediction_type == None:
            if self.dependent_col in self.df._get_numeric_data().columns.tolist():
                self.prediction_type = 'regression'
            else:
                self.prediction_type = 'classification'
        else:
            self.prediction_type = prediction_type

    def _set_categorical_columns(self, categorical_cols):
        #TODO: Need to test if the columns exist in the df
        #TODO: Add logic in case user doesn't specify the cols
        if categorical_cols == None:
            num_cols = self.df._get_numeric_data().columns
            cat_cols = list(set(self.df.columns) - set(num_cols) - set([self.dependent_col]))
            self.categorical_cols = cat_cols
            ## empty list in case of no categorical columns or categorical columns are
            ## already label encoded, as in the case of Randal's data.
            ## Assumption: In case of pre-processed data, all columns would be preprocessed
            ## and not just some. Hence, proceed with heuristics only if previous code 
            ## gave zero categorical_cols
            # print cat_cols
            if cat_cols == []:
                possible_cat_cols = []
                threshold_unique = 0.001*self.df.shape[0]
                # print threshold_unique
                for col in list(set(self.df.columns) - set([self.dependent_col])):
                    unique_col = list(self.df[col].unique())
                    unique_col.sort()
                    # print col, len(unique_col)
                    if len(unique_col) < threshold_unique:
                        possible_cat_cols.append(col)
                        continue
                    # print unique_col == range(0, len(unique_col), 1)
                    # print  isinstance(self.df[col][0], np.integer)
                    # If unique values represent intergers from 0 to N, then there
                    # is a high chance they were LabelEncoded using sklearn.
                    # This heaveily relies on the way experiment datasets were encoded.
                    # Not recommended for normal usage.
                    
                    if ((unique_col == range(0, len(unique_col), 1)) & (isinstance(self.df[col][0], np.integer))):   
                        possible_cat_cols.append(col)
                        continue
                self.categorical_cols = list(set(possible_cat_cols))
        else:
            self.categorical_cols = categorical_cols

    
    def _categorical_column_encoder(self):
        """ Assumes all categorical variables are nominal and not
        ordinal """
        categorical_cols = self.categorical_cols
        
        self.df_encoded = self.df.copy()

        for col in categorical_cols:
            if len(self.df_encoded[col].unique())<=2:
                #this means, binary :- LabelEncode
                self.df_encoded[col] = LabelEncoder().fit_transform(self.df_encoded[col])
            else:
                # nominal - so make dummy"
                self.df_encoded = pd.get_dummies(self.df_encoded, columns=[col])
        

    def n_rows(self):
        return self.df.shape[0]

    def n_columns(self):
        """ Including dependent variable """
        return self.df.shape[1]

    def ratio_rowcol(self):
        """ rows/col including dependent variable """
        return self.df.shape[0]/self.df.shape[1]

    def n_categorical(self):
        """number of categorical variables excluding the dependent."""
        #todo: can be converted to ratio by total number of columns.
        return len(self.categorical_cols)

    def n_numerical(self):
        """number of categorical variables excluding the dependent."""
        #todo: can be converted to ratio by total number of columns.
        return self.n_columns() - self.n_categorical() - 1        

    def n_classes(self):
        """number of classes in the dependent columns. Only applicable
        for classfication problems. Returns NaN otherwise """

        if self.prediction_type == 'classification':
            return len(self.df[self.dependent_col].unique())
        else:
            return np.nan

    ## Post-encoding dimensional stats.
    #todo: n_cols_post_encoding
    #todo: ratop_rowcol_post_encoding


    #----------------------------------------------------------------------
    # Correlation related
    corr_with_dependent = None

    def _get_corr_with_dependent(self):
        """Called from init. Sets up data for correlation related meta-features.
        #todo: take-call - Should I make different classes/modules for
        different types of meta-features? Eg. Correlation, Entropy"""
        
        #Correlation with dependent variable only make sense for regression problems
        if self.prediction_type == 'regression':
            if self.corr_with_dependent!=None:
                return self.corr_with_dependent
            else:
                self.corr_with_dependent = self.df_encoded.corr()[self.dependent_col]
                self.corr_with_dependent = self.corr_with_dependent.loc[self.corr_with_dependent.index!=self.dependent_col]
                return self.corr_with_dependent

    def corr_with_dependent_abs_max(self):
        """ max absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return abs_corr_with_dependent.max()
    
    def corr_with_dependent_abs_min(self):
        """ min absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return abs_corr_with_dependent.min()    


    def corr_with_dependent_abs_mean(self):
        """ mean absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return abs_corr_with_dependent.mean()    

    def corr_with_dependent_abs_median(self):
        """ median absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return abs_corr_with_dependent.median()    



    def corr_with_dependent_abs_std(self):
        """ std absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return abs_corr_with_dependent.std(ddof = 1)    



    def corr_with_dependent_abs_25p(self):
        """ 25p absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return np.nanpercentile(abs_corr_with_dependent, 25)   



    def corr_with_dependent_abs_75p(self):
        """ 75p absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return np.nanpercentile(abs_corr_with_dependent, 75)

    #todo: try kurtosis and skew for correl values without abs.

    def corr_with_dependent_abs_kurtosis(self):
        """ kurtosis of absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        from scipy.stats import kurtosis
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return kurtosis(abs_corr_with_dependent, bias = False)

    def corr_with_dependent_abs_skew(self):
        """ skew of absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            abs_corr_with_dependent = self._get_corr_with_dependent().abs()
            return skew(abs_corr_with_dependent, bias = False)

    #----------------------------------------------------------------------
    # Class probablity related
    class_probablities = None
    def _get_class_probablity(self):
        if self.class_probablities is None:
            dependent_col = self.df[self.dependent_col]
            class_counts = dependent_col.value_counts()
            self.class_probablities = class_counts/self.n_rows()
            return self.class_probablities
        else:
            return self.class_probablities

    def class_prob_min(self):
        if self.prediction_type=='regression':
            return np.nan
        else:
            class_probablities = self._get_class_probablity()
            return class_probablities.min()
    
    def class_prob_max(self):
        if self.prediction_type=='regression':
            return np.nan
        else:
            class_probablities = self._get_class_probablity()
            return class_probablities.max()
    
    def class_prob_std(self):
        if self.prediction_type=='regression':
            return np.nan
        else:
            class_probablities = self._get_class_probablity()
            return class_probablities.std(ddof = 1)    
    
    def class_prob_mean(self):
        if self.prediction_type=='regression':
            return np.nan
        else:
            class_probablities = self._get_class_probablity()
            return class_probablities.mean()    
    
    def class_prob_median(self):
        if self.prediction_type=='regression':
            return np.nan
        else:
            class_probablities = self._get_class_probablity()
            return class_probablities.median()

    #todo: add kurtosis and skew here too. Classes will be usually less, so 
    #may not make sense.


    #----------------------------------------------------------------------
    # Symbols related - All the categorical columns

    symbol_counts_dict = None
    def _get_symbols_per_category(self):
        """
        Sets an dictionary with number of symbols per categorical 
        column using categorical_cols info.
        """


        if self.symbol_counts_dict == None:
            self.symbol_counts_dict = {}
            for column in self.categorical_cols:
                self.symbol_counts_dict[column] = self.df[column].dropna().unique().shape[0]
            return self.symbol_counts_dict
        else:
            return self.symbol_counts_dict

    def symbols_mean(self):
        """ Average symbols per columns """

        symbol_counts_dict = self._get_symbols_per_category()
        ## None is for checking empty, no categorical columns
        
        if not symbol_counts_dict:
            return np.nan
        symbol_counts = symbol_counts_dict.values()

        return np.nanmean(symbol_counts)


    def symbols_std(self):
        """ std of symbols per columns """
        symbol_counts_dict = self._get_symbols_per_category()
        ## None is for checking empty, no categorical columns
        if not symbol_counts_dict:
            return np.nan
        symbol_counts = symbol_counts_dict.values()

        return np.nanstd(symbol_counts, ddof = 1)

    
    def symbols_min(self):
        """ Average symbols per columns """
        symbol_counts_dict = self._get_symbols_per_category()
        ## None is for checking empty, no categorical columns
        if not symbol_counts_dict:
            return np.nan
        symbol_counts = symbol_counts_dict.values()

        return np.min(symbol_counts)   

    def symbols_max(self):
        """ Average symbols per columns """
        symbol_counts_dict = self._get_symbols_per_category()
        ## None is for checking empty, no categorical columns

        if not symbol_counts_dict:
            return np.nan
        symbol_counts = symbol_counts_dict.values()

        return np.max(symbol_counts)

    def symbols_sum(self):
        """ Sum of symbols per column """
        symbol_counts_dict = self._get_symbols_per_category()
        ## None is for checking empty, no categorical columns
        if not symbol_counts_dict:
            return np.nan

        symbol_counts = symbol_counts_dict.values()

        return np.sum(symbol_counts)  

    def symbols_skew(self):
        from scipy.stats import skew
        symbol_counts_dict = self._get_symbols_per_category()
        ## None is for checking empty, no categorical columns
        if not symbol_counts_dict:
            return np.nan

        symbol_counts = symbol_counts_dict.values()

        return skew(symbol_counts, bias = False)  

    def symbols_kurtosis(self):
        from scipy.stats import kurtosis
        symbol_counts_dict = self._get_symbols_per_category()
        ## None is for checking empty, no categorical columns
        if not symbol_counts_dict:
            return np.nan

        symbol_counts = symbol_counts_dict.values()

        return kurtosis(symbol_counts, bias = False)  

    


    ##todo: Note we can evaluate symbol probabilities too.

