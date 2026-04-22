from classifier.classifier import VesselClassifier
from nlp_engine.feature_extractor import FeatureExtractor
from nlp_engine.text_preprocessor import TextPreprocessor


def main():
    print("Начало работы программы")

    print("Инициализация Текстового препроцессора")
    text_preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    classifier = VesselClassifier()

    # Тестовые примеры
    test_cases = [
        "Круизное морское судно для перевозки пассажиров",
        "Морской танкер для транспортировки сырой нефти, дедвейт 150000 тонн",
        "Атомный ледокол для прокладки маршрутов во льдах Арктики",
        "Контейнеровоз для перевозки стандартизированных контейнеров по мировому океану",
        "Пассажирско-автомобильный паром для прибрежных перевозок",
        "Научно-исследовательское судно для океанографических измерений без груза"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        # Тестирование иерархического подхода (по умолчанию)
        print("\n" + "="*60)
        print(f"ТЕСТ {i}: ИЕРАРХИЧЕСКИЙ ПОДХОД (use_hierarchical=True)")
        print("="*60)
        result_hierarchical = classifier.classify(test_text, use_hierarchical=True)
        classifier.show_result(result_hierarchical)

        # Тестирование классического подхода
        print("\n" + "="*60)
        print(f"ТЕСТ {i}: КЛАССИЧЕСКИЙ ПОДХОД (use_hierarchical=False)")
        print("="*60)
        result_classic = classifier.classify(test_text, use_hierarchical=False)
        classifier.show_result(result_classic)


if __name__ == '__main__':
    main()