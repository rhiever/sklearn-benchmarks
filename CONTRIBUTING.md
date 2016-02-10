# Contributing

## How to contribute

The preferred way to contribute to sklearn-benchmarks is to fork the 
[main repository](https://github.com/rhiever/sklearn-benchmarks/) on
GitHub:

1. Fork the [project repository](https://github.com/rhiever/sklearn-benchmarks):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourLogin/sklearn-benchmarks.git
          $ cd sklearn-benchmarks

3. Create a branch to hold your changes:

          $ git checkout -b my-contribution

   and start making changes. Never work in the ``master`` branch!

4. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-contribution

Finally, go to the web page of your fork of the sklearn-benchmarks repo,
and click 'Pull request' to send your changes to the maintainers for
review. This will send an email to the maintainers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

## Contributing data

If you would like to add a data set to these benchmarks, make sure that they meet the following criteria:

* The data set must be in a single file. If the data set comes with a training and testing set, combine the two sets together into a single file. We will be performing our own divisions of the data set for cross-validation.

* The data set must be in standard CSV format, i.e., with commas as column separators. The rows must represent an entry in the data set, and the columns must represent the features and class. **Ensure that `pandas` can read in the file** with the `pandas.read_csv('filename')` command.

* The first row in the data set must be the column names. The class column must be named `class` for consistency across the analysis.

* The data set must contain entirely numerical features/classes. All categorical features/classes must be converted to numerical counterparts.

* The data file must be zipped with `gzip --best` to ensure that it takes up as little disk space as possible

Feel free to browse the [existing data sets](https://github.com/rhiever/sklearn-benchmarks/tree/master/data) in this benchmark to get an idea of what we expect. Once your data set meets all of the above criteria, you can contribute it to the `data` directory via a pull request as discussed in the "How to contribute" section.
