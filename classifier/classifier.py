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

    def classify(self, text: str) -> dict:
        """Классификация текстового описания"""
        return self.inference_engine.classify(text, self.feature_extractor)

    @staticmethod
    def show_result(result) -> None:
        print(f"Оригинальный текст: {result['original_text']}")
        print(f"Признаки: \n"
              f"    raw_terms: {result['features']['raw_terms']}\n"
              f"    vessel_type: {result['features']['vessel_type']}\n"
              f"    attributes: {result['features']['attributes']}\n")

        print(f"Лучшее правило: ")
        print(f"{result.get('best_match')}")

        print(f"Правила: ")
        [print(rule) for rule in result['matched_rules']]


    def get_classification_tree(self, text: str):
        """Получение иерархии классификации"""
        ...