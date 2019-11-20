#!/usr/bin/env python
#
#  Author: Adam Jakab
#  Copyright: Copyright (c) 2019., Adam Jakab
#  License: See LICENSE.txt
#  Email: adaja at itu dot dk
#


from text_cat.text_cat import TextCat

csv1 = 'Data/mini-dataset.csv'
csv2 = 'Data/example-dataset.csv'

tc = TextCat("Naive Model Experiment #1", csv2)
# tc.train_naive_bayes_classifier()
tc.train_sklearn_classifier()
# tc.classifier.show_most_informative_features(15)


toks = [
    "This was the stupidest movie ever!",
    "Good job! What a surprise!",
    "This was only a start I hope.",
    "sensitive and astute first feature",
    "it was just as fun as an aching tooth",
    "it was funny"
]
for tok in toks:
    print("'{0}' gets label: {1}".format(tok, tc.classify_sentence(tok)))


