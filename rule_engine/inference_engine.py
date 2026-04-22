from typing import List, Dict, Any, Optional, Tuple

from nlp_engine.feature_extractor import FeatureExtractor
from rule_engine.rule_matcher import RuleMatcher


class InferenceEngine:
    """Движок логического вывода для классификации судов"""
    def __init__(self, rules: List[dict]):
        self.rule_matcher = RuleMatcher()
        self.rule_matcher.load_rules(rules)
        
        # Индексируем правила для быстрого доступа
        self.rules_by_code = {rule['code']: rule for rule in rules}
        self.rules_by_parent = {}
        self.root_rules = []
        
        for rule in rules:
            parent_code = rule.get('parent_code')
            if parent_code is None:
                self.root_rules.append(rule)
            else:
                if parent_code not in self.rules_by_parent:
                    self.rules_by_parent[parent_code] = []
                self.rules_by_parent[parent_code].append(rule)

        self.cache = {}

    def _find_best_match_in_level(self, rules: List[dict], features: dict) -> Optional[Tuple[dict, float]]:
        """Находит лучшее правило среди списка правил одного уровня"""
        best_match = None
        best_ratio = 0.0
        
        for rule in rules:
            is_matched, match_ratio = self.rule_matcher.evaluate_rule(rule, features)
            
            # Учитываем только правила с match_ratio > 0.5 или полностью совпавшие
            if is_matched or match_ratio > 0.5:
                if match_ratio > best_ratio:
                    best_ratio = match_ratio
                    best_match = rule
                elif match_ratio == best_ratio and best_match is not None:
                    # При одинаковом match_ratio выбираем более специфичное (с большим уровнем)
                    if rule['level'] > best_match['level']:
                        best_match = rule
        
        if best_match is not None:
            return (best_match, best_ratio)
        return None

    def _classify_hierarchical(self, features: dict) -> List[Dict[str, Any]]:
        """
        Поэтапная классификация по иерархии правил.
        Возвращает путь классификации от корня до листа.
        """
        classification_path = []
        
        # Начинаем с правил первого уровня
        current_level_rules = self.root_rules
        
        while current_level_rules:
            # Ищем лучшее правило на текущем уровне
            best_result = self._find_best_match_in_level(current_level_rules, features)
            
            if best_result is None:
                # Нет подходящих правил на этом уровне - останавливаемся
                break
            
            best_rule, match_ratio = best_result
            
            # Добавляем найденное правило в путь классификации
            classification_path.append({
                'code': best_rule['code'],
                'name': best_rule['name'],
                'match_ratio': match_ratio,
                'parent_code': best_rule.get('parent_code'),
                'level': best_rule['level'],
                'description': best_rule['rule'].get('description', ''),
                'rule': best_rule['rule'],
            })
            
            # Переходим к потомкам этого правила
            current_level_rules = self.rules_by_parent.get(best_rule['code'], [])
        
        return classification_path

    def classify(self, text: str, feature_extractor: FeatureExtractor, use_hierarchical: bool = True) -> dict:
        """
        Классификация текстового описания судна.
        
        Args:
            text: Текстовое описание судна
            feature_extractor: Извлекатель признаков
            use_hierarchical: Если True - используется поэтапная иерархическая классификация,
                             если False - классификация по всем правилам сразу
        """

        if text in self.cache:
            cached_result = self.cache[text]
            # Возвращаем кэшированный результат, но учитываем текущий режим
            if use_hierarchical:
                return cached_result
            else:
                # Для не-иерархического режима пересчитываем только matched_rules
                features = feature_extractor.extract_features(text)
                matched_rules = self.rule_matcher.match_rules(features)
                
                result = {
                    'original_text': text,
                    'features': features,
                    'matched_rules': [],
                    'best_match': None,
                    'confidence': 0.0,
                    'alternatives': [],
                    'classification_path': []
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
                    best_rule, best_ratio = max(matched_rules, key=lambda x: x[1])
                    result['best_match'] = {
                        'code': best_rule['code'],
                        'name': best_rule['name'],
                        'match_ratio': best_ratio,
                        'parent_code': best_rule.get('parent_code'),
                        'level': best_rule['level'],
                        'description': best_rule['rule'].get('description', '')
                    }
                    result['confidence'] = best_ratio
                    
                    all_others = [r for r in matched_rules if r[0]['code'] != best_rule['code']]
                    result['alternatives'] = [
                        {
                            'code': rule['code'],
                            'name': rule['name'],
                            'match_ratio': ratio
                        }
                        for rule, ratio in all_others[:5]
                        if ratio >= 0.7
                    ]
                
                return result

        features = feature_extractor.extract_features(text)

        if use_hierarchical:
            # Выполняем поэтапную иерархическую классификацию
            classification_path = self._classify_hierarchical(features)
        else:
            classification_path = []

        # Получаем все совпавшие правила
        matched_rules = self.rule_matcher.match_rules(features)

        result = {
            'original_text': text,
            'features': features,
            'matched_rules': [],
            'best_match': None,
            'confidence': 0.0,
            'alternatives': [],
            'classification_path': classification_path
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

        if use_hierarchical and classification_path:
            # Лучшее правило - последнее в пути классификации (самое специфичное)
            best_in_path = classification_path[-1]
            
            result['best_match'] = {
                'code': best_in_path['code'],
                'name': best_in_path['name'],
                'match_ratio': best_in_path['match_ratio'],
                'parent_code': best_in_path.get('parent_code'),
                'level': best_in_path['level'],
                'description': best_in_path.get('description', '')
            }
            result['confidence'] = best_in_path['match_ratio']

            # Собираем альтернативы из остальных совпавших правил
            best_code = best_in_path['code']
            all_others = [r for r in matched_rules if r[0]['code'] != best_code]
            result['alternatives'] = [
                {
                    'code': rule['code'],
                    'name': rule['name'],
                    'match_ratio': ratio

                }
                for rule, ratio in all_others[:5]
                if ratio >= 0.7
            ]
        elif not use_hierarchical and matched_rules:
            # Классический подход: лучшее правило по match_ratio
            best_rule, best_ratio = max(matched_rules, key=lambda x: x[1])
            
            result['best_match'] = {
                'code': best_rule['code'],
                'name': best_rule['name'],
                'match_ratio': best_ratio,
                'parent_code': best_rule.get('parent_code'),
                'level': best_rule['level'],
                'description': best_rule['rule'].get('description', '')
            }
            result['confidence'] = best_ratio
            
            all_others = [r for r in matched_rules if r[0]['code'] != best_rule['code']]
            result['alternatives'] = [
                {
                    'code': rule['code'],
                    'name': rule['name'],
                    'match_ratio': ratio
                }
                for rule, ratio in all_others[:5]
                if ratio >= 0.7
            ]

        if use_hierarchical:
            self.cache[text] = result
            
        return result




    def get_classification_tree(self, result: dict) -> List[dict]:
        """Получение иерархии классификации"""
        ...