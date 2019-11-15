
from text_cat.text_cat import TextCat

csv1 = 'Data/mini-dataset.csv'
csv2 = 'Data/example-dataset.csv'

tc = TextCat(csv2)
tc.train_naive_bayes_classifier()
tc.classifier.show_most_informative_features(15)

tok = "This was the stupidest movie ever!"
label = tc.classify_sentence(tok)
print("'{0}' get label: {1}".format(tok, label))



