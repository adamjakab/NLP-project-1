
import re
from random import shuffle
import pickle
import nltk
# from nltk.classify.scikitlearn import SklearnClassifier
# from textblob.classifiers import NaiveBayesClassifier


class TextCat:
    def __init__(self, csv_file, featureset_size=1000, test_ratio=0.3, feature_ratio=0.1):
        self.csv_file = csv_file
        self.documents = []
        self.words = []
        self.featureset_size = 1
        self.test_ratio = test_ratio
        self.feature_ratio = feature_ratio
        self.feature_words = None
        self.classifier = None
        print("TC ready.")


    def _read_csv(self):
        with open(self.csv_file, 'r') as input_csv:
            for item in input_csv:
                item = item.split(',')
                doc, label = re.findall('\\w+', ''.join(item[:-1]).lower()), item[-1].strip()
                for word in doc:
                    if len(word) > 2:
                        self.words.append(word.lower())
                self.documents.append((doc, label))
            unique_word_count = len(list(set(self.words)))
            self.featureset_size = int(unique_word_count * self.feature_ratio)
            if self.featureset_size > 500:
                self.featureset_size = 500

            print("Unique Word Count: {0}".format(unique_word_count))
            print("Featureset Size(r: {0}): {1}".format(self.feature_ratio, self.featureset_size))


    def _generate_word_features(self):
        frequency_dist = nltk.FreqDist()
        for word in self.words:
            frequency_dist[word] += 1

        # totally random - as they were read in
        # self.feature_words = list(frequency_dist)[:self.featureset_size]

        # ordered by the most common
        self.feature_words = [tok for (tok, cnt) in frequency_dist.most_common(self.featureset_size)]
        print(self.feature_words)


    def __document_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.feature_words:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    def train_naive_bayes_classifier(self):
        if not self.feature_words:
            self._read_csv()
            self._generate_word_features()

        shuffle(self.documents)
        feature_sets = [(self.__document_features(tok), lab) for (tok, lab) in self.documents]
        # print(feature_sets)

        cutoff = int(len(feature_sets) * self.test_ratio)
        train_set, test_set = feature_sets[cutoff:], feature_sets[:cutoff]
        print("Totals({0}) Training({1}) Test({2})".format(len(feature_sets), len(train_set), len(test_set)))

        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

        print('Achieved {0:.2f}% accuracy against training set'.format(
            nltk.classify.accuracy(self.classifier, train_set) * 100))
        print('Achieved {0:.2f}% accuracy against test set'.format(
            nltk.classify.accuracy(self.classifier, test_set) * 100))


    def save_model(self, filename):
        save_classifier = open(filename, "wb")
        pickle.dump(self.classifier, save_classifier)
        save_classifier.close()
        save_vocab = open('vocab-{}'.format(filename), "wb")
        pickle.dump(self.feature_words, save_vocab)
        save_vocab.close()

    def load_model(self, model_filename, vocab_filename):
        classifier_f = open(model_filename, "rb")
        self.classifier = pickle.load(classifier_f)
        classifier_f.close()
        vocab_f = open(vocab_filename, "rb")
        self.feature_words = pickle.load(vocab_f)
        vocab_f.close()


    def classify_sentence(self, sentence):
        if not self.feature_words:
            self._read_csv()
            self._generate_word_features()
        test_features = {}
        for word in self.feature_words:
            test_features['contains({})'.format(word.lower())] = (word.lower() in nltk.word_tokenize(sentence))
        return self.classifier.classify(test_features)

