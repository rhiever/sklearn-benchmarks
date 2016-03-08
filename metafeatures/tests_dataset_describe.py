import unittest
import pandas as pd
import numpy as np
import math
from dataset_describe import Dataset
 
class Dataset_Describe(unittest.TestCase):
 
    def setUp(self):
        # classification problem.
        iris = pd.read_csv('iris.csv')
        self.iris = Dataset(iris)   

        # Regression problem
        tips = pd.read_csv('tips.csv')
        self.tips = Dataset(tips, dependent_col = 'tip')   


    def test_number_of_rows(self):
        result = self.iris.n_rows()
        self.assertEqual(150, result)

    def test_number_of_columns(self):
        result = self.iris.n_columns()
        self.assertEqual(5, result)
 
    def test_number_of_categorical_vars(self):
        result = self.iris.n_categorical()
        self.assertEqual(0, result)

    def test_number_of_numerical_vars(self):
        result = self.iris.n_numerical()
        self.assertEqual(4, result)

    def test_total_nclasses(self):
        result = self.iris.n_classes()
        self.assertEqual(3, result)

    def test_total_nclasses_in_regression_problem(self):
        result = self.tips.n_classes()
        self.assertTrue(math.isnan(result))
    
    def test_prediction_type_classification(self):
        result = self.iris.prediction_type
        self.assertEqual('classification', result)

    def test_prediction_type_regression(self):
        result = self.tips.prediction_type
        self.assertEqual('regression', result)


    def test_max_corr_with_dependent_classification(self):
        result = self.iris.max_abs_corr_with_dependent()
        self.assertTrue(math.isnan(result))

    def test_max_corr_with_dependent_regression(self):
        result = self.tips.max_abs_corr_with_dependent()
        self.assertAlmostEqual(0.675,result, places = 2)

    def test_min_corr_with_dependent_regression(self):
        result = self.tips.min_abs_corr_with_dependent()
        self.assertAlmostEqual(0.002, result, places = 2)



if __name__ == '__main__':
    unittest.main()