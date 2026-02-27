import pandas as pd
import numpy as np
from collections import defaultdict
import random

# -------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# -------------------------------------------------
def load_data(ratings_path, movies_path, test_ratio=0.2, seed=42):
    # Загрузка рейтингов
    ratings = pd.read_csv(
        ratings_path,
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    # Загрузка фильмов
    movies = pd.read_csv(
        movies_path,
        sep='::',
        names=['movie_id', 'title', 'genres'],
        encoding='latin1',
        engine='python'
    )
    movie_id_to_title = dict(zip(movies['movie_id'], movies['title']))
    
    # Разделение на train/test
    if test_ratio > 0:
        test_size = int(len(ratings) * test_ratio)
        ratings_shuffled = ratings.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_ratings = ratings_shuffled.iloc[test_size:].copy()
        full_ratings = ratings.copy()
    else:
        train_ratings = ratings.copy()
        full_ratings = ratings.copy()
    
    return train_ratings, full_ratings, movie_id_to_title

# -------------------------------------------------
# 2. ПОСТРОЕНИЕ СЛОВАРЯ ФИЛЬМ -> ПОЛЬЗОВАТЕЛИ
# -------------------------------------------------
def build_item_ratings(df):
    item_ratings = defaultdict(dict)
    for _, row in df.iterrows():
        item_ratings[row['movie_id']][row['user_id']] = row['rating']
    return dict(item_ratings)

# -------------------------------------------------
# 3. КОСИНУСНОЕ СХОДСТВО
# -------------------------------------------------
def cosine_sim(movie_a, movie_b, item_ratings):
    ra = item_ratings.get(movie_a, {}) # Вектор оценок первого фильма
    rb = item_ratings.get(movie_b, {}) # Вектор оценок второго фильма
    common = set(ra.keys()) & set(rb.keys()) # Пересечение пользователей у фильмов

    if not common:
        return 0.0
    
    dot = sum(ra[u] * rb[u] for u in common) # Скалярное произведение векторов
    norm_a = np.sqrt(sum(v**2 for v in ra.values())) # Нормализация вектора(длина) по ВСЕМ оценкам, точнее направление вектора
    norm_b = np.sqrt(sum(v**2 for v in rb.values())) # Нормализация вектора(длина)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b) # Косинусное сходство между оценками по 1 поль

# -------------------------------------------------
# 4. ТОП-X ПОХОЖИХ ФИЛЬМОВ
# -------------------------------------------------
def get_top_similar(target_movie, item_ratings, all_movies, top_k=10):
    sims = []
    for mid in all_movies:
        if mid == target_movie:
            continue
        s = cosine_sim(target_movie, mid, item_ratings)
        if s > 0:
            sims.append((mid, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in sims[:top_k]]

# -------------------------------------------------
# 5. JACCARD INDEX ДЛЯ ОЦЕНКИ КАЧЕСТВА
# -------------------------------------------------
def jaccard_index(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

def evaluate_jaccard_quality(train_item_ratings, full_item_ratings, movie_id_to_title, top_k=10, n_samples=100):
    all_movies = list(train_item_ratings.keys())
    sample_movies = random.sample(all_movies, min(n_samples, len(all_movies)))
    
    jaccards = []
    for mid in sample_movies:
        recs_train = set(get_top_similar(mid, train_item_ratings, all_movies, top_k))
        recs_full = set(get_top_similar(mid, full_item_ratings, list(full_item_ratings.keys()), top_k))
        jaccards.append(jaccard_index(recs_train, recs_full))
    
    return np.mean(jaccards)
# -------------------------------------------------
# 6. ВЫВОД СПИСКОВ
# -------------------------------------------------
def print_recommendations(movie_id, rec_list, id2title, label="Рекомендации"):
    print(f"\n{label}:")
    for i, mid in enumerate(rec_list, 1):
        title = id2title.get(mid, f"[ID {mid}]")
        print(f"  {i}. {title}")
# -------------------------------------------------
# 7. ОСНОВНАЯ ПРОГРАММА
# -------------------------------------------------
if __name__ == "__main__":
    # Параметры (можно сделать через input или argparse)
    RATING_FILE = "data/ratings.dat"
    MOVIE_FILE = "data/movies.dat"
    TEST_RATIO = 0.2
    TOP_K = 10
    N_SAMPLES_FOR_JACCARD = 100

    print("Загрузка данных...")
    train_df, full_df, id2title = load_data(RATING_FILE, MOVIE_FILE, test_ratio=TEST_RATIO)

    print("Построение моделей...")
    train_item_ratings = build_item_ratings(train_df)
    full_item_ratings = build_item_ratings(full_df)

    print("Оценка качества (Jaccard Index)...")
    jaccard_score = evaluate_jaccard_quality(
        train_item_ratings, full_item_ratings, id2title,
        top_k=TOP_K, n_samples=N_SAMPLES_FOR_JACCARD
    )
    print(f"Средний Jaccard Index @Top-{TOP_K}: {jaccard_score:.4f}\n")

    all_movies_train = list(train_item_ratings.keys())
all_movies_full = list(full_item_ratings.keys())

print("Готово! Введите ID фильма для сравнения рекомендаций.")
print(f"(Пример популярного фильма: 1 — 'Toy Story')\n")

while True:
    try:
        user_input = input("Введите ID фильма (или 'quit' для выхода): ").strip()
        if user_input.lower() == 'quit':
            break
        movie_id = int(user_input)
        
        if movie_id not in id2title:
            print("Фильм с таким ID не найден.")
            continue

        # Получаем рекомендации от обеих моделей
        recs_train = get_top_similar(movie_id, train_item_ratings, all_movies_train, top_k=TOP_K)
        recs_full = get_top_similar(movie_id, full_item_ratings, all_movies_full, top_k=TOP_K)

        # Выводим названия
        print(f"\n🎯 Оригинальный фильм: '{id2title[movie_id]}'")
        print_recommendations(movie_id, recs_train, id2title, f"На 80% данных (train), top-{TOP_K}:")
        print_recommendations(movie_id, recs_full, id2title, f"На 100% данных (full), top-{TOP_K}:")

        # Считаем и выводим Jaccard для этого фильма
        j_single = jaccard_index(set(recs_train), set(recs_full))
        print(f"\n📊 Jaccard Index для этого фильма: {j_single:.3f}")

    except ValueError:
        print("Пожалуйста, введите целое число.")
    except KeyboardInterrupt:
        break

    print("До свидания!")