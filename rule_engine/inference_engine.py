from msilib import Feature
from typing import List

from nlp_engine.feature_extractor import FeatureExtractor
from rule_engine.rule_matcher import RuleMatcher


class InferenceEngine:
    """Движок логического вывода для классификации судов"""
    def __init__(self, rules: List[dict]):
        self.rule_matcher = RuleMatcher()
        self.rule_matcher.load_rules(rules)

        self.cache = {}

    def classify(self, text: str, feature_extractor: FeatureExtractor) -> dict:
        """Классификация текстового описания судна"""

        if text in self.cache:
            return self.cache[text]

        features = feature_extractor.extract_features(text)

        matched_rules = self.rule_matcher.match_rules(features)

        result = {
            'original_text': text,
            'features': features,
            'matched_rules': [],
            'best_matched': None,
            'confidence': 0.0,
            'alternatives': []
        }

        for rule, match_ratio in matched_rules:
            rule_results = {
                'code': rule['code'],
                'name': rule['name'],
                'match_ratio': match_ratio,
                'parent_code': rule.get('parent_code'),
                'level': rule['level'],
                'description': rule['rule'].get('description', ''),
                'rule': rule['rule'],
            }
            result['matched_rules'].append(rule_results)

        if matched_rules:

            max_ratio = max(r[1] for r in matched_rules)

            best_by_ratio = [r for r in matched_rules if r[1] == max_ratio]

            max_level = max(rule['level'] for rule, _ in best_by_ratio)

            most_specific = [
                (rule, ratio) for rule, ratio in best_by_ratio
                if rule['level'] == max_level
            ]

            best_rule, best_ratio = most_specific[0]

            result['best_match'] = {
                'code': best_rule['code'],
                'name': best_rule['name'],
                'match_ratio': best_ratio,
                'parent_code': best_rule.get('parent_code'),
                'level': best_rule['level'],
                'description': best_rule['rule'].get('description', '')
            }
            result['confidence'] = best_ratio

            all_others = [r for r in matched_rules if r != (best_rule, best_ratio)]
            result['alternatives'] = [
                {
                    'code': rule['code'],
                    'name': rule['name'],
                    'match_ratio': ratio

                }
                for rule, ratio in all_others[:5]
                if ratio >= 0.7
            ]

            if len(most_specific) > 1:
                extra_alternatives = [
                    {
                        'code': rule['code'],
                        'name': rule['name'],
                        'match_ratio': ratio
                    }
                    for rule, ratio in most_specific[1:]
                ]
                result['alternatives'] = extra_alternatives + result['alternatives']

        self.cache[text] = result
        return result




    def get_classification_tree(self, result: dict) -> List[dict]:
        """Получение иерархии классификации"""
        ...