import hashlib
import re
from random import shuffle
import pickle
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.decisiontree import DecisionTreeClassifier
# from textblob.classifiers import NaiveBayesClassifier


class TextCat:
    model_name = None

    def __init__(self, model_name, csv_file, test_ratio=0.2, feature_ratio=0.1):
        self.model_name = model_name
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
            if self.featureset_size > 100:
                self.featureset_size = 100

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
        # print(self.feature_words)


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

        self.__save_model()

    def train_sklearn_classifier(self):
        if not self.feature_words:
            self._read_csv()
            self._generate_word_features()

        shuffle(self.documents)
        feature_sets = [(self.__document_features(tok), lab) for (tok, lab) in self.documents]

        cutoff = int(len(feature_sets) * self.test_ratio)
        train_set, test_set = feature_sets[cutoff:], feature_sets[:cutoff]
        print("Totals({0}) Training({1}) Test({2})".format(len(feature_sets), len(train_set), len(test_set)))

        self.classifier = DecisionTreeClassifier.train(test_set)

        print('Achieved {0:.2f}% accuracy against training set'.format(
            nltk.classify.accuracy(self.classifier, train_set) * 100))
        print('Achieved {0:.2f}% accuracy against test set'.format(
            nltk.classify.accuracy(self.classifier, test_set) * 100))


    def __save_model(self):
        model_cache_name = self.__get_model_cache_name()
        filename_classifier = "Cache/{0}.classifier.cache".format(model_cache_name)
        filename_vocab = "Cache/{0}.vocab.cache".format(model_cache_name)

        # Classifier
        save_classifier = open(filename_classifier, "wb+")
        pickle.dump(self.classifier, save_classifier)
        save_classifier.close()

        # Vocab
        save_vocab = open(filename_vocab, "wb+")
        pickle.dump(self.feature_words, save_vocab)
        save_vocab.close()

    def load_model(self):
        model_cache_name = self.__get_model_cache_name()
        filename_classifier = "Cache/{0}.classifier.cache".format(model_cache_name)
        filename_vocab = "Cache/{0}.vocab.cache".format(model_cache_name)

        # Classifier
        classifier_f = open(filename_classifier, "rb")
        self.classifier = pickle.load(classifier_f)
        classifier_f.close()

        vocab_f = open(filename_vocab, "rb")
        self.feature_words = pickle.load(vocab_f)
        vocab_f.close()

    def __get_model_cache_name(self):
        md5 = hashlib.md5()
        clean_model_name = str(self.model_name).strip().lower().replace(' ', '_')
        clean_model_name = re.sub(r'(?u)[^-\w.]', '', clean_model_name)
        md5.update(clean_model_name.encode('UTF-8'))
        return md5.hexdigest()

    def classify_sentence(self, sentence):
        if not self.classifier or not self.feature_words:
            self.load_model()

        test_features = {}
        for word in self.feature_words:
            test_features['contains({})'.format(word.lower())] = (word.lower() in nltk.word_tokenize(sentence))

        return self.classifier.classify(test_features)

