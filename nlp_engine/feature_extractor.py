from typing import Any, List, Dict

from .text_preprocessor import TextPreprocessor


class FeatureExtractor:
    """Движок по извлечению признаков из предварительно обработанного текста"""

    def __init__(self):

        self.preprocessor = TextPreprocessor()

        """Основные"""
        # Зона плавания
        # река - море
        self.term_to_navigation_area = {
            "река_море": "река_море",
            "река_мор": "река_море",
            "морской": "морское",
            "море": "морское",
            "океан": "морское",
            "речной": "речное",
        }
        # Назначение
        # Военное,
        # пассажирское
        # Грузовое
        # Рыболовное
        # Буксировка
        # Прочее
        # Спорт
        # Прогулочное
        self.term_to_purpose = {
            "спасательный": "спасательный",
            "аварийно_спасательный": "аварийно_спасательный",
            "военное": "военное",
            "пассажирское": "пассажирское",
            "грузовое": "грузовое",
            "грузоперевозка": "грузовое",
            "рыболовное": "рыболовное",
            "буксировка": "буксировка",
            "спорт": "спорт",
            "прогулочное": "прогулочное",
            "прогулочный": "прогулочное",
            "прочее": "прочее",
        }

        # тип груза
        # люди, пассажиры
        # Нефть, сжиженный газ, нефтепродукты, химические продукты, жидкий
        # Контейнер, Трейлеры, Навалочный, Лес, Комбинированные
        # Железнодорожные, Автомобильно-транспортные "Паромы морские самоходные железнодорожные, автомобильно-транспортные"
        self.term_to_cargo_type = {
            "сжиженный_газ": "сжиженный_газ",
            "пассажир": "пассажиры",
            "нефть": "нефть",
            "танкер_сырцевик": "нефть",
            "нефтепродукты": "нефтепродукты",
            "химические_продукты": "химические_продукты",
            "жидкий": "жидкий",
            "контейнер": "контейнер",
            "трейлер": "трейлер",
            "навалочный": "навалочный",
            "зерновой": "навалочный",
            "лес": "лес",
            "комбинированный": "комбинированный",
            "железнодорожные": "железнодорожные",
            "автомобильно_транспортные": "автомобильно_транспортные",
            "автомобиль": "автомобильно_транспортные",
            "авто": "автомобильно_транспортные",

        }

        # Специализация
        # экскурсионное, круизное, паром
        # Рыбозавод, Траулер, Дрифтер, Сейнер, Ярусник, Китобойное, Зверобойное,
        # Добыча, вылов краба, Консервирование,
        # Рейдовый, Портовый, спасательный,
        # Промерное, Навигационно-гидрографическое, Буровое, Многофункциональное вспомогательное, Лоцмейстерское
        # Обслуживающий флот, Снабжение
        # понтон Для плавучего крана
        # ракеты, космос
        self.term_to_specialization = {
            "сжиженный_газ": "газовоз",
            "газовоз": "газовоз",
            "экскурсионное": "экскурсионное",
            "круизное": "круизное",
            "круиз": "круизное",
            "круизный": "круизное",
            "паром": "паром",
            "рыбозавод": "рыбозавод",
            "траулер": "траулер",
            "дрифтер": "дрифтер",
            "сейнер": "сейнер",
            "ярусник": "ярусник",
            "китобойное": "китобойное",
            "зверобойное": "зверобойное",
            "краб": "краб",
            "консервирование": "консервирование",
            "рейдовое": "рейдовое",
            "портовый": "портовый",
            "спасательный": "спасательный",
            "промерное": "промерное",
            "навигационно_гидрографическое": "навигационно_гидрографическое",
            "буровое": "буровое",
            "снабжение": "снабжение",
            "лоцмейстерское": "лоцмейстерское",
            "понтон": "понтон",
            "ракета": "ракета",
            "космос": "космос",
        }
        # Тип судна
        # Корабль, судно, лодка, плавучая конструкция, плавучие средства
        # пассажирское
        # Наливное, Танкер, Газовоз, Рефрижератор
        # Сухогруз, Балкер
        # Грузопассажирское (Объединение грузового и пассажирского)
        # Рыболовное, суда-рабозаподы
        # Буксир, Толкач, Судно-толкач, Буксир-толкач, Катер буксирный
        # Земснаряд, плавучий маяк, плавучий кран, Пожарное,
        # Прочие (Для узкоспециализированного 30.11.33)
        # Платформа, плот, понтон, кессон, дебаркадер, буи, бакен
        # Прогулочные, спортивные
        self.term_to_vessel_type = {
            "корабль": "Ship",
            "лодка": "Boat",
            "плавучая_конструкция": "FloatingStructure",
            "плавуче_средство": "FloatingStructure",
            "пассажирское": "Ship",
            "пассажир": "Ship",
            "круиз": "Ship",
            "круизное": "Ship",
            "наливное": "Ship",
            "газовоз": "Ship",
            "танкер": "Ship",
            "танкер_сырцевик": "Ship",
            "рефрижератор": "Ship",
            "сухогруз": "Ship",
            "контейнер": "Ship",
            "балкер": "Ship",
            "грузопассажирское": "Ship",
            "рыболовное": "Ship",
            "рыбозавод": "Ship",
            "буксир": "Ship",
            "буксир_толкач": "Ship",
            "толкач": "Ship",
            "земснаряд": "Ship",
            "маяк": "FloatingStructure",
            "кран": "FloatingStructure",
            "пожарное": "Ship",
            "прочее": "Vessel",
            "платформа": "Platform",
            "плот": "FloatingStructure",
            "понтон": "FloatingStructure",
            "кессон": "FloatingStructure",
            "дебаркадер": "FloatingStructure",
            "буи": "FloatingStructure",
            "бакен": "FloatingStructure",
            "прогулочное": "Boat",
            "спортивное": "Boat",
            "ледокол": "Ship",
            "военное": "Ship",
            "судно": "Vessel"
        }

        """
            Пока непонятно
            Суда для обслуживания регулярных пассажирских линий морские
            Суда морские пассажирские прочие
            Суда морские для перевозки химических продуктов
            Суда сухогрузные морские прочие
        """
        """
            Самоходное, Несамоходное, вспомогательный двигатель или без него
        """
        """
            Услуги, переоборудование, восстановление, оснащение, операции, производство, субподрядчик
        """
        """
            Парусное, надувное, прогулочное, спортивное, гребное, шлюпка, каноэ, лодка, дежурные
        """
        """Дополнительные"""
        self.term_to_tug_type = {}
        self.term_to_propulsion_type = {}
        self.term_to_structure_type = {}


        # Плавучие, погружные, космическая, причал
        self.term_to_platform_type = {}
        self.term_to_boolean_attributes = {}
        self.term_to_construction_type = {}


    def extract_vessel_type(self, terms: List[str]) -> str:

        for term in self.term_to_vessel_type:
            if term in terms:
                return self.term_to_vessel_type[term]

        return 'Boat'


    def extract_attributes(self, terms: List[str]) -> Dict[str, Any]:
        """Извлекаем аттрибуты из терминов"""
        attributes = {}

        # Извлечение зоны плавания
        for term in self.term_to_navigation_area:
            if term in terms:
                attributes['navigation_area'] = self.term_to_navigation_area[term]
                break

        for term in self.term_to_propulsion_type:
            if term in terms:
                attributes['propulsion_type'] = self.term_to_propulsion_type[term]
                break

        for term in self.term_to_purpose:
            if term in terms:
                attributes['purpose'] = self.term_to_purpose[term]
                break

        # Извлечение типа груза
        for term in self.term_to_cargo_type:
            if term in terms:
                attributes['cargo_type'] = self.term_to_cargo_type[term]
                break



        # Извлечение специализации
        for term in self.term_to_specialization:
            if term in terms:
                attributes['specialization'] = self.term_to_specialization[term]
                break

        return attributes

    @staticmethod
    def extract_numbers(numbers_dict: Dict[str, float]) -> Dict[str, float]:
        """Извлечение числовых характеристик"""
        numeric_attributes = {}

        if 'тоннаж' in numbers_dict:
            numeric_attributes['tonnage'] = numbers_dict['тоннаж']

        if 'вместимость' in numbers_dict:
            numeric_attributes['passenger_capacity'] = numbers_dict['вместимость']

        if 'длина' in numbers_dict:
            numeric_attributes['length'] = numbers_dict['длина']

        if 'грузоподъемность' in numbers_dict:
            numeric_attributes['cargo_capacity'] = numbers_dict['грузоподъемность']

        if 'грузоподъемность' in numbers_dict:
            numeric_attributes['deadweight'] = numbers_dict['грузоподъемность']

        return numeric_attributes


    def extract_features(self, text: str) -> Dict[str, Any]:
        """Полное извлечение признаков из текста"""
        # Предварительная обработка
        preprocessed = self.preprocessor.preprocess(text)

        vessel_type = self.extract_vessel_type(preprocessed['normalized_terms'])
        attributes = self.extract_attributes(preprocessed['normalized_terms'])
        numeric_attributes = self.extract_numbers(preprocessed['numbers'])

        features = {
            'original_text': text,
            'raw_terms': preprocessed['normalized_terms'],
            'vessel_type': vessel_type,
            'attributes': attributes,
            'numeric_attributes': numeric_attributes
        }
        return features






