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
    categorical_col = None
    dependent_col = None
    prediction_type = None
    independent_col = None
    def __init__(self, df, prediction_type = None, dependent_col = None,categorical_col = None):
        
        self.df = df
        self._set_dependent_col(dependent_col)
        self._set_categorical_columns(categorical_col)
        self._set_prediction_type(prediction_type)

        self.independent_col = list(set(self.df.columns.tolist()) - set(self.dependent_col))
        self._categorical_column_encoder()

        # setups
        self._setup_correlations()

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

    def _set_categorical_columns(self, categorical_col):
        #TODO: Need to test if the columns exist in the df
        #TODO: Add logic in case user doesn't specify the cols
        if categorical_col == None:
            num_cols = self.df._get_numeric_data().columns
            cat_cols = list(set(self.df.columns) - set(num_cols) - set([self.dependent_col]))
            self.categorical_col = cat_cols
        else:
            self.categorical_col = categorical_col

    
    def _categorical_column_encoder(self):
        """ Assumes all categorical variables are nominal and not
        ordinal """
        categorical_cols = self.categorical_col
        
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
        return len(self.categorical_col)

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


    ## Correlation related measures
    corr_with_dependent = None
    abs_corr_with_dependent = None

    def _setup_correlations(self):
        """Called from init. Sets up data for correlation related meta-features.
        #todo: take-call - Should I make different classes/modules for
        different types of meta-features? Eg. Correlation, Entropy"""
        
        #Correlation with dependent variable only make sense for regression problems
        if self.prediction_type == 'regression':
            self.corr_with_dependent = self.df_encoded.corr()[self.dependent_col]
            self.corr_with_dependent = self.corr_with_dependent.loc[self.corr_with_dependent.index!=self.dependent_col]
            self.abs_corr_with_dependent = self.corr_with_dependent.abs()
    
    def max_abs_corr_with_dependent(self):
        """ max absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            return self.abs_corr_with_dependent.max()
    
    def min_abs_corr_with_dependent(self):
        """ min absolute pearson correlation with dependent variable
        returns np.nan for classificaiton problems. Uses df_encoded
        ie dataframe with categorical columns encoded automatically.
        """
        if self.prediction_type == 'classification':
            return np.nan
        else:
            return self.abs_corr_with_dependent.min()

    ## Entropy related measures.

