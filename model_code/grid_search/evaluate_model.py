import sys
import itertools
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from tpot_metrics import balanced_accuracy_score
import warnings

def evaluate_model(dataset, pipeline_components, pipeline_parameters):
    input_data = pd.read_csv(dataset, compression='gzip', sep='\t')
    features = input_data.drop('class', axis=1).values.astype(float)
    labels = input_data['class'].values

    pipelines = [dict(zip(pipeline_parameters.keys(), list(parameter_combination)))
                 for parameter_combination in itertools.product(*pipeline_parameters.values())]

    with warnings.catch_warnings():
        # Squash warning messages. Turn this off when debugging!
        warnings.simplefilter('ignore')

        for pipe_parameters in pipelines:
            pipeline = []
            for component in pipeline_components:
                if component in pipe_parameters:
                    args = pipe_parameters[component]
                    pipeline.append(component(**args))
                else:
                    pipeline.append(component())

            try:
                clf = make_pipeline(*pipeline)
                cv_predictions = cross_val_predict(estimator=clf, X=features, y=labels, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=90483257))
                accuracy = accuracy_score(labels, cv_predictions)
                macro_f1 = f1_score(labels, cv_predictions, average='macro')
                balanced_accuracy = balanced_accuracy_score(labels, cv_predictions)
            except KeyboardInterrupt:
                sys.exit(1)
            # This is a catch-all to make sure that the evaluation won't crash due to a bad parameter
            # combination or bad data. Turn this off when debugging!
            except Exception as e:
                continue

            classifier_class = pipeline_components[-1]
            param_string = ','.join(['{}={}'.format(parameter, value)
                                    for parameter, value in pipe_parameters[classifier_class].items()])

            out_text = '\t'.join([dataset.split('/')[-1][:-7],
                                classifier_class.__name__,
                                param_string,
                                str(accuracy),
                                str(macro_f1),
                                str(balanced_accuracy)])

            print(out_text)
            sys.stdout.flush()
