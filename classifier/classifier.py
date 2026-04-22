from cnl_rules.rules_generator import rules_reader
from nlp_engine.feature_extractor import FeatureExtractor
from rule_engine.inference_engine import InferenceEngine


class VesselClassifier:

    def __init__(self):
        self.rules = rules_reader.get_rules()

        self.feature_extractor = FeatureExtractor()
        self.inference_engine = InferenceEngine(self.rules)


    def load_all_rules(self):
        raise NotImplementedError

    def classify(self, text: str, use_hierarchical: bool = True) -> dict:
        """
        Классификация текстового описания.
        
        Args:
            text: Текстовое описание судна
            use_hierarchical: Если True - используется поэтапная иерархическая классификация,
                             если False - классификация по всем правилам сразу
        """
        return self.inference_engine.classify(text, self.feature_extractor, use_hierarchical)

    @staticmethod
    def show_result(result) -> None:
        print(f"Оригинальный текст: {result['original_text']}")
        print(f"Признаки: \n"
              f"    raw_terms: {result['features']['raw_terms']}\n"
              f"    vessel_type: {result['features']['vessel_type']}\n"
              f"    attributes: {result['features']['attributes']}\n")

        print(f"Лучшее правило: ")
        print(f"{result.get('best_match')}")

        # Вывод пути классификации (поэтапная иерархия)
        classification_path = result.get('classification_path', [])
        if classification_path:
            print(f"\nПуть классификации (иерархия):")
            for i, step in enumerate(classification_path, 1):
                print(f"  Уровень {i}: {step['code']} - {step['name']} (совпадение: {step['match_ratio']:.2f})")

        print(f"\nВсе совпавшие правила: ")
        [print(rule) for rule in result['matched_rules']]


    def get_classification_tree(self, text: str) -> list:
        """Получение иерархии классификации"""
        result = self.classify(text)
        return result.get('classification_path', [])