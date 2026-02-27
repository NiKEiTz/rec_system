# Item-Based Рекомендательная система (MovieLens 1M)

Реализация рекомендательной системы на основе схожести фильмов (item-based collaborative filtering) без использования готовых библиотек ML.

## 💡 Что делает система?
- По ID фильма выдаёт топ-N похожих фильмов.
- Использует **косинусное сходство** между векторами оценок.
- Оценивает качество через **Jaccard Index** между рекомендациями на train и full данных.

## 📥 Данные
Используется датасет [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).  
Скачайте архив, распакуйте и положите файлы `ratings.dat` и `movies.dat` в папку `data/`.

Структура: data/ratings.dat - оценки, data/movies.dat - фильмы.


## ▶️ Запуск
1. Установите зависимости:
   ```bash
   pip install pandas numpy

2. Запустите скрипт:
    python recommender.py

3. Пример вывода:
    Средний Jaccard Index @Top-10: 0.6241

    Введите ID фильма (или 'quit' для выхода): 1

    Рекомендации для фильма: 'Toy Story (1995)'
        1. Toy Story 2 (1999)
        2. Monsters, Inc. (2001)
        3. Finding Nemo (2003)

