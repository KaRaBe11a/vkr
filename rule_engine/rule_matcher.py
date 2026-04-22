import json
import operator
from typing import List, Tuple


class RuleMatcher:
    """Движок сопоставления правил CNL с извлеченными признаками"""

    OPERATORS = {
        'equals': operator.eq,
        'not_equals': operator.ne,
        'in': lambda a, b: a in b if isinstance(b, (list, set, tuple)) else a == b,
        'not_in': lambda a, b: a not in b if isinstance(b, (list, set, tuple)) else a != b,
        'greater_than': operator.gt,
        'less_than': operator.lt,
        'greater_than_or_equal': operator.ge,
        'less_than_or_equal': operator.le,
        'contains': lambda a, b: b in a if isinstance(a, (str, list, set, tuple)) else False,
        'not_contains': lambda a, b: b not in a if isinstance(a, (str, list, set, tuple)) else True,
        'exists': lambda a, b: a is not None and a != '',
        'not_exists': lambda a, b: a is None or a == '',
        'count_greater_than': lambda a, b: len(a) > b if isinstance(a, (list, set, tuple, dict)) else False
    }


    def __init__(self):
        self.rules = []
        self.loaded = False

    def load_rules_from_file(self, filepath: str):
        """Загрузка правил из JSON файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
            self.rules = rules_data if isinstance(rules_data, list) else [rules_data]
        self.loaded = True

    def load_rules(self, rules: List[dict]):
        """Загрузка правил из списка"""
        self.rules = rules
        self.loaded = True

    def evaluate_condition(self, condition: dict, features: dict) -> bool:
        """Оценка одного условия"""
        try:
            attr_name = condition['attribute']

            attr_value = None

            if 'attributes' in features and attr_name in features['attributes']:
                attr_value = features['attributes'][attr_name]
            elif attr_name == "is_instance_of":
                # Сначала проверяем vessel_type
                attr_value = features.get('vessel_type')
                
                # Если правило требует "пассажирское" или "Пассажирское", 
                # также проверяем атрибут purpose как fallback
                expected_value = condition.get('value')
                if expected_value and isinstance(expected_value, str):
                    expected_lower = expected_value.lower()
                    if expected_lower in ['пассажирское', 'passengerboat', 'passenger']:
                        # Проверяем, есть ли в attributes purpose='пассажирское'
                        if 'attributes' in features and features['attributes'].get('purpose') == 'пассажирское':
                            attr_value = 'пассажирское'

            if attr_value is None:
                if condition['operator'] in ['not_exists', 'not_equals', 'not_in']:
                    return True
                return False


            op_name = condition['operator']
            op_func = self.OPERATORS.get(op_name)

            if not op_func:
                raise ValueError(f"Неизвестный оператор: {op_name}")

            expected_value = condition.get('value')
            
            # Нормализуем сравнение для строковых значений (регистронезависимое)
            if isinstance(attr_value, str) and isinstance(expected_value, str):
                # Для is_instance_of с значением "Пассажирское" сравниваем регистронезависимо
                if attr_name == 'is_instance_of':
                    if attr_value.lower() == expected_value.lower():
                        return True
            
            # Для оператора 'in' проверяем, является ли attr_value подтипом ожидаемого значения
            # Например, PassengerBoat должен совпадать с Ship, Boat, Vessel
            if op_name == 'in' and attr_name == 'is_instance_of':
                if isinstance(expected_value, (list, set, tuple)):
                    # Проверяем точное совпадение
                    if attr_value in expected_value:
                        return True
                    # Проверяем иерархию типов
                    type_hierarchy = {
                        'PassengerBoat': 'Ship',
                        'Tanker': 'Ship',
                        'GasCarrier': 'Ship',
                        'RefrigeratorShip': 'Ship',
                        'DryCargoBoat': 'Ship',
                        'ContainerShip': 'Ship',
                        'BulkCarrier': 'Ship',
                        'FishingBoat': 'Ship',
                        'FishingFactory': 'Ship',
                        'TugBoat': 'Ship',
                        'Dredger': 'Ship',
                        'SpecialBoat': 'Ship',
                        'TimberCarrier': 'Ship',
                        'Longliner': 'Ship',
                        'Seiner': 'Ship',
                        'Trawler': 'Ship',
                        'Whaler': 'Ship',
                        'Ferry': 'Ship',
                        'RecreationalBoat': 'Boat',
                    }
                    base_type = type_hierarchy.get(attr_value)
                    if base_type and base_type in expected_value:
                        return True
                    # Также проверяем Vessel как корневой тип для всех судов
                    if attr_value in type_hierarchy and 'Vessel' in expected_value:
                        return True
            
            result = op_func(attr_value, expected_value)
            return result
        except Exception as e:
            print(f"Ошибка при оценке условия: {e}")
            return False


    def evaluate_logic(self, logic_block: dict, features: dict) -> bool:
        """Оценка логического блока AND OR"""
        logic_type = logic_block.get('logic', 'AND')
        conditions = logic_block.get('conditions', [])

        if logic_type == "AND":
            return all(self.evaluate_condition(cond, features) for cond in conditions)
        elif logic_type == "OR":
            return any(self.evaluate_condition(cond, features) for cond in conditions)
        else:
            raise ValueError(f"Неизвестный тип логики: {logic_type}")


    def evaluate_rule(self, rule: dict, features: dict) -> Tuple[bool, float]:
        """Оценка правил и расчёт степени соответствия"""
        try:
            conditions = rule['rule']['conditions']
            matched_conditions = 0
            total_conditions = len(conditions)

            for condition in conditions:
                if 'logic' in condition:
                    result = self.evaluate_logic(condition, features)
                else:
                    result = self.evaluate_condition(condition, features)

                if result:
                    matched_conditions += 1

            match_ratio = matched_conditions / total_conditions if total_conditions > 0 else 0

            is_matched = match_ratio == 1.0

            return is_matched, match_ratio

        except Exception as e:
            print(f"Ошибка при оценке правила {rule.get('code')}: {e}")
            return False, 0.0

    def match_rules(self, features: dict) -> List[Tuple[dict, float]]:
        """Сопоставление признаков с правилами"""
        if not self.loaded:
            raise ValueError("Правила не загружены")

        matched_rules = []

        for rule in self.rules:
            is_matched, match_ratio = self.evaluate_rule(rule, features)

            if is_matched or match_ratio > 0.5:
                matched_rules.append((rule, match_ratio))

        matched_rules.sort(key=lambda x: x[1], reverse=True)

        return matched_rules

