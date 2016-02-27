""" 
Methods to describe attriubutes in a dataset. The last column
of a dataset is assumed to be the dependent variable. 

Methods range from but not restricted to:
 - Description of dataset as a whole
 - Description of individual attributes
 - Description of inter-relation between attributes.

"""

import pandas as pd

class Dataset:
    """
    Initialize the dataset and give user the option to set some
    defaults eg. names of categorical columns

    All public methods will provide one value per dataset.
    Private methods are for internal use.  

    """
    df = None
    categorical_col = None


    def __init__(self, df, categorical_col = None):
        
        self.df = df
        self._set_categorical_columns(categorical_col)

    def _set_categorical_columns(self, categorical_col):
        #TODO: Need to test if the columns exist in the df
        #TODO: Add logic in case user doesn't specify the cols
        self.categorical_col = categorical_col

    
    def nrows(self):
        return self.df.shape[0]

    def ncolumns(self):
       return self.df.shape[1]

    def ratio_rowcol(self):
       return self.df.shape[0]/self.df.shape[1]

    def ncategorical(self):
       """number of categorical variables"""
       return len(self.categorical_col)