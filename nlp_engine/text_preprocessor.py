import re
from typing import Tuple, Any, List, Set, Dict

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from config import settings


class TextPreprocessor:
    """Предварительная обработка текста, для извлечения ключевых терминов"""

    def __init__(self):
        self.morph = MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))
        self.stop_words.update(
            [
                'быть', 'ходить' # Сюда можно добавить стоп слова которые будут удаляться что бы не мешать
            ]
        )

        # Словарь составных терминов (биграммы и триграммы)
        self.composite_lemmas = {
            'вспомогательный двигатель': 'вспомогательный_двигатель',
            'сжиженный газ': "сжиженный_газ",
            'природный газ': 'природный_газ',
            'химический продукт': 'химический_продукт',
            'буксир толкач': 'буксир_толкач',

            'буксир аварийно_спасательный': 'буксир_аварийно_спасательный',
            'буксир аварийный спасательный': 'буксир_аварийно_спасательный',
            'сжидить природный газ': 'сжиженный_газ',
            'сжидить нефтяной газ': 'сжиженный_газ',
        }

        # Словарь синонимов (одиночные термины)
        self.synonyms = {}


    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        # Проверить может есть более умные токенизаторы текста
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = text.replace("-", "_")
        tokens = text.split()
        return tokens

    def extract_composite_terms(self, lemmas: List[str]) -> Tuple[List[str], Set[str]]:
        """
        Извлечение составных терминов из списка лемм.
        Берём слова по тройкам/двойкам и смотрим есть ли такая двойка/тройка в словаре биграм/триграм
        """
        normalized_composites = []
        used_words = set()
        i = 0

        while i < len(lemmas):
            matched = False

            if i + 2 < len(lemmas):
                trigram = ' '.join(lemmas[i:i + 3])
                if trigram in self.composite_lemmas:
                    normalized_composites.append(self.composite_lemmas[trigram])
                    used_words.update(lemmas[i:i + 3])
                    i += 3
                    matched = True

            if not matched and i + 1 < len(lemmas):
                bigram = ' '.join(lemmas[i:i + 2])
                if bigram in self.composite_lemmas:
                    normalized_composites.append(self.composite_lemmas[bigram])
                    used_words.update(lemmas[i:i + 2])
                    i += 2
                    matched = True

            if not matched:
                i += 1
        return normalized_composites, used_words


    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Удаление стоп слов и слишком коротких токенов"""
        filtered_tokens = [
            token for token in tokens
            if token not in self.stop_words
               and len(token) >= settings.min_len_token
        ]
        return filtered_tokens

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Лемматизация токенов"""
        lemmas = []
        for token in tokens:
            parsed = self.morph.parse(token)[0]
            lemma = parsed.normal_form
            lemmas.append(lemma)
        return lemmas


    def normalize_synonyms(self, lemmas: List[str], exclude_words: Set[str] = None) -> List[str]:
        """
        Нормализация синонимов (для одиночных терминов)
        :param lemmas:
        :param exclude_words: Множество слов которые уже вошли в составные термины
        :return:
        """
        if exclude_words is None:
            exclude_words = set()

        normalized = []
        for lemma in lemmas:
            if lemma in exclude_words:
                continue

            found = False
            for main_term, synonyms in self.synonyms.items():
                for tmp_lemma in synonyms:
                    normalized.append(tmp_lemma)
                    found = True
                    break
            if not found:
                normalized.append(lemma)
        return normalized


    @staticmethod
    def extract_numbers(text: str) -> Dict[str, float]:
        """Извлечение числовых значений из текста"""
        numbers = {}

        tonnage_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:тонн|т)', text, re.IGNORECASE)
        if tonnage_match:
            numbers['тоннаж'] = float(tonnage_match.group(1))

        capacity_match = re.search(r'(\d+)\s*(?:пассажиров|пасс|чел)', text, re.IGNORECASE)
        if capacity_match:
            numbers['вместимость'] = float(capacity_match.group(1))

        length_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:метров|м)', text, re.IGNORECASE)
        if length_match:
            numbers['длина'] = float(length_match.group(1))

        cargo_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:грузоподъемность|грузоподъемностью)', text, re.IGNORECASE)
        if cargo_match:
            numbers['грузоподъемность'] = float(cargo_match.group(1))

        return numbers


    def preprocess(self, text) -> Dict[str, Any]:
        """Полная предварительная обработка текста"""
        tokens = self.tokenize(text)
        numbers = self.extract_numbers(text)

        filtered_tokens = self.remove_stopwords(tokens)

        lemmas = self.lemmatize(filtered_tokens)

        composite_terms, used_words = self.extract_composite_terms(lemmas)

        normalized = self.normalize_synonyms(lemmas, exclude_words=used_words)

        all_terms = composite_terms + normalized

        return {
            'original_text': text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'lemmas': lemmas,
            'composite_terms': composite_terms,
            'normalized_terms': all_terms,
            'numbers': numbers,
            'used_words': used_words,
        }



