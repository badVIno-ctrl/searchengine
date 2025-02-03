import asyncio
import string
import time
from functools import lru_cache
from collections import defaultdict
import heapq
import math
from typing import List, Dict, Tuple, Optional
from pymorphy3 import MorphAnalyzer
from symspellpy import SymSpell, Verbosity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download
import logging
import uuid

from nltk.stem import WordNetLemmatizer



from data import texts
from config import (
    STOPWORDS,
    CACHE_SIZE,
    SYNONYMS,
    BM25_K1,
    BM25_B,
    MAX_EDIT_DISTANCE,
    PHRASE_SEARCH_WINDOW,
    USER_WEIGHT,
    RUSSIAN_STOPWORDS,
    ENGLISH_STOPWORDS,
    MULTILINGUAL_STEMMING
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncSearchEngine:
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.index = defaultdict(dict)
        self.doc_info = []
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.stopwords = RUSSIAN_STOPWORDS.union(ENGLISH_STOPWORDS).union(STOPWORDS)
        self.sym_spell = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE)
        self.users = {}

        self._init_nltk()
        self._build_index()
        self._build_spell_checker()

        self.index_lock = asyncio.Lock()
        self.user_lock = asyncio.Lock()

    def _init_nltk(self):
        try:
            nltk_download(['stopwords', 'wordnet', 'omw-1.4'], quiet=True)
        except Exception as e:
            logger.error(f"NLTK init error: {e}")
            raise

    def _preprocess_sync(self, text: str) -> List[str]:
        words = []
        for word in text.translate(str.maketrans('', '', string.punctuation)).lower().split():
            if len(word) > 2:
                if MULTILINGUAL_STEMMING:
                    if any(cyrillic in word for cyrillic in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
                        lemma = self.morph.parse(word)[0].normal_form
                    else:
                        lemma = self.lemmatizer.lemmatize(word)
                else:
                    lemma = word

                if lemma not in self.stopwords:
                    words.append(lemma)
        return words

    def __init__(self):
        ...
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = RUSSIAN_STOPWORDS.union(ENGLISH_STOPWORDS).union(STOPWORDS)
        ...

    def _preprocess_sync(self, text: str) -> List[str]:
        words = []
        for word in text.translate(str.maketrans('', '', string.punctuation)).lower().split():
            if len(word) > 2:
                # Определение языка слова
                if any(cyrillic in word for cyrillic in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
                    lemma = self.morph.parse(word)[0].normal_form
                else:
                    lemma = self.lemmatizer.lemmatize(word)
                if lemma not in self.stopwords:
                    words.append(lemma)
        return words

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchMetrics:

    def __init__(self):
        self.query_times = []
        self.clickthrough_rates = defaultdict(float)
        self.relevance_scores = defaultdict(list)

    def log_query_time(self, time: float):
        self.query_times.append(time)

    def update_relevance(self, query: str, doc_id: int, score: int):
        self.relevance_scores[query].append((doc_id, score))

    def get_avg_query_time(self) -> float:
        return sum(self.query_times) / len(self.query_times) if self.query_times else 0

    def get_precision(self, query: str) -> float:
        ...


class UserProfile:

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.search_history = defaultdict(int)
        self.click_history = defaultdict(int)

    def update_search_history(self, terms: List[str]):
        for term in terms:
            self.search_history[term] += 1

    def update_click_history(self, doc_id: int):
        self.click_history[doc_id] += 1


class AsyncSearchEngine:

    def __init__(self):
        self.morph = MorphAnalyzer()
        self.index: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
        self.doc_info: List[Dict] = []
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.stopwords: set = set()
        self.sym_spell = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE)
        self.metrics = SearchMetrics()
        self.users: Dict[str, UserProfile] = {}

        self._init_nltk()
        self._build_index()
        self._build_spell_checker()

        self.index_lock = asyncio.Lock()
        self.user_lock = asyncio.Lock()

    def _init_nltk(self):
        try:
            nltk_download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('russian')).union(STOPWORDS)
        except Exception as e:
            logger.error(f"NLTK init error: {e}")
            raise

    async def _preprocess(self, text: str) -> List[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: [
                self.morph.parse(word)[0].normal_form
                for word in text.translate(
                    str.maketrans('', '', string.punctuation)
                ).lower().split()
                if len(word) > 2 and self.morph.parse(word)[0].normal_form not in self.stopwords
            ]
        )

    def _build_index(self):
        for doc_id, text in enumerate(texts):
            words = self._preprocess_sync(text)
            self.doc_info.append({
                'tf': defaultdict(int),
                'positions': defaultdict(list),
                'length': len(words),
                'original': text
            })

            for pos, word in enumerate(words):
                self.index[word].setdefault(doc_id, []).append(pos)
                self.doc_info[doc_id]['tf'][word] += 1

        self.total_docs = len(texts)
        self.avg_doc_length = sum(d['length'] for d in self.doc_info) / self.total_docs

    def _preprocess_sync(self, text: str) -> List[str]:
        return [
            self.morph.parse(word)[0].normal_form
            for word in text.translate(
                str.maketrans('', '', string.punctuation)
            ).lower().split()
            if len(word) > 2 and self.morph.parse(word)[0].normal_form not in self.stopwords
        ]

    def _build_spell_checker(self):
        for word in self.index:
            self.sym_spell.create_dictionary_entry(word, 1)

    async def _correct_spelling(self, word: str) -> str:
        suggestions = self.sym_spell.lookup(
            word,
            Verbosity.TOP,
            max_edit_distance=MAX_EDIT_DISTANCE
        )
        return suggestions[0].term if suggestions else word

    async def _process_query(self, query: str) -> List[str]:
        terms = await self._preprocess(query)
        corrected = []
        for term in terms:
            corrected.append(await self._correct_spelling(term))
        return corrected

    async def _phrase_search(self, terms: List[str]) -> List[int]:
        doc_matches = defaultdict(list)
        for i, term in enumerate(terms):
            async with self.index_lock:
                doc_positions = self.index.get(term, {})

            for doc_id, positions in doc_positions.items():
                for pos in positions:
                    doc_matches[doc_id].append(pos - i)

        return [
            doc_id
            for doc_id, offsets in doc_matches.items()
            if any(offsets.count(offset) == len(terms) for offset in offsets)
        ]

    async def _bm25_score(self, doc_id: int, terms: List[str], user: Optional[UserProfile] = None) -> float:
        score = 0.0
        doc = self.doc_info[doc_id]

        for term in terms:
            if term not in self.index:
                continue

            doc_freq = len(self.index[term])
            idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            tf = doc['tf'].get(term, 0)
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc['length'] / self.avg_doc_length)
            score += idf * (numerator / denominator)

            if user:
                score += USER_WEIGHT * math.log(1 + user.search_history.get(term, 0))
                score += USER_WEIGHT * math.log(1 + user.click_history.get(doc_id, 0))

        return score

    @lru_cache(maxsize=CACHE_SIZE)
    async def search(self, query: str, user_id: Optional[str] = None) -> Tuple[List[str], Dict]:
        start_time = time.time()

        user = await self._get_user(user_id) if user_id else None

        terms = await self._process_query(query)
        if not terms:
            return [], {}

        phrase_docs = await self._phrase_search(terms)

        expanded_terms = await self._expand_query(terms)

        doc_scores = defaultdict(float)
        async with self.index_lock:
            for term in expanded_terms:
                for doc_id in self.index.get(term, {}).keys():
                    doc_scores[doc_id] += await self._bm25_score(doc_id, expanded_terms, user)

        for doc_id in phrase_docs:
            doc_scores[doc_id] *= 2.0

        results = heapq.nlargest(5, doc_scores.items(), key=lambda x: x[1])

        self.metrics.log_query_time(time.time() - start_time)
        if user:
            async with self.user_lock:
                user.update_search_history(terms)

        return [self.doc_info[doc_id]['original'] for doc_id, _ in results], {
            'corrected_query': " ".join(terms),
            'metrics': {
                'processing_time': time.time() - start_time,
                'documents_found': len(doc_scores)
            }
        }

    async def _expand_query(self, terms: List[str]) -> List[str]:
        expanded = set(terms)
        for term in terms:
            expanded.update(SYNONYMS.get(term, []))
        return list(expanded)

    async def _get_user(self, user_id: str) -> UserProfile:
        async with self.user_lock:
            if user_id not in self.users:
                self.users[user_id] = UserProfile(user_id)
            return self.users[user_id]


async def main():
    engine = AsyncSearchEngine()
    user_id = str(uuid.uuid4())

    while True:
        try:
            query = input("Введите запрос (exit для выхода): ").strip()
            if query.lower() == 'exit':
                break

            results, meta = await engine.search(query, user_id)
            print(f"\nРезультаты для '{meta['corrected_query']}':")
            print(f"Время обработки: {meta['metrics']['processing_time']:.2f}s")
            for i, text in enumerate(results[:3], 1):
                print(f"\n{i}. {text[:150]}...")

        except Exception as e:
            logger.error(f"Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
