import json
import os
from typing import List


class RulesReader:
    """Менеджер читающий все правила из json файла"""
    def __init__(self):
        self._rules = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.rules_file = os.path.normpath(os.path.join(current_dir, "..", "data", "cnl_rules_full.json"))

    def load_rules_from_file(self) -> List[dict] :
        try:
            if not os.path.exists(self.rules_file):
                raise FileNotFoundError(f"Файл правил не найден")

            with open(self.rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._rules = data.get("rules")

            print(f"Загружено {len(self._rules)} правил из файла {self.rules_file}")
            return self._rules
        except Exception as e:
            print(f"Ошибка при загрузке правил: {e}")
            return []

    def get_rules(self) -> List[dict]:
        if not self._rules:
            self.load_rules_from_file()
        return self._rules


rules_reader = RulesReader()
