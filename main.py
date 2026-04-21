from classifier.classifier import VesselClassifier
from nlp_engine.feature_extractor import FeatureExtractor
from nlp_engine.text_preprocessor import TextPreprocessor


def main():
    print("Начало работы программы")

    print("Инициализация Текстового препроцессора")
    text_preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    classifier = VesselClassifier()

    result = classifier.classify("Круизное морское судно для перевозки пассажиров")
    # [print(res) for res in result.items()]
    classifier.show_result(result)



if __name__ == '__main__':
    main()