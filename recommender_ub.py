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
        test_ratings = ratings_shuffled.iloc[:test_size].copy()
    else:
        train_ratings = ratings.copy()
        test_ratings = pd.DataFrame()
    
    return train_ratings, test_ratings, movie_id_to_title

# -------------------------------------------------
# 2. ПОСТРОЕНИЕ СЛОВАРЯ ПОЛЬЗОВАТЕЛЬ -> ФИЛЬМЫ
# -------------------------------------------------
def build_user_ratings(df):
    """Возвращает: {user_id: {movie_id: rating}}"""
    user_ratings = defaultdict(dict)
    for _, row in df.iterrows():
        user_ratings[row['user_id']][row['movie_id']] = row['rating']
    return dict(user_ratings)

# -------------------------------------------------
# 3. КОСИНУСНОЕ СХОДСТВО МЕЖДУ ПОЛЬЗОВАТЕЛЯМИ
# -------------------------------------------------
def cosine_sim_users(user_a, user_b, user_ratings):
    """
    Считает сходство между двумя пользователями
    на основе фильмов, которые оценили оба
    """
    ratings_a = user_ratings.get(user_a, {})
    ratings_b = user_ratings.get(user_b, {})
    
    # Общие фильмы
    common_movies = set(ratings_a.keys()) & set(ratings_b.keys())
    if not common_movies:
        return 0.0
    
    # Скалярное произведение
    dot = sum(ratings_a[mid] * ratings_b[mid] for mid in common_movies)
    
    # Нормы (длины векторов)
    norm_a = np.sqrt(sum(v**2 for v in ratings_a.values()))
    norm_b = np.sqrt(sum(v**2 for v in ratings_b.values()))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)

# -------------------------------------------------
# 4. НАХОЖДЕНИЕ ТОП-K ПОХОЖИХ ПОЛЬЗОВАТЕЛЕЙ
# -------------------------------------------------
def get_top_similar_users(target_user, user_ratings, all_users, top_k=10):
    """
    Возвращает список [(похожий_пользователь, сходство), ...]
    """
    similarities = []
    for user in all_users:
        if user == target_user:
            continue
        sim = cosine_sim_users(target_user, user, user_ratings)
        if sim > 0:  # Только положительное сходство
            similarities.append((user, sim))
    
    # Сортируем по убыванию сходства
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# -------------------------------------------------
# 5. РЕКОМЕНДАЦИЯ ФИЛЬМОВ (ОСНОВНАЯ ФУНКЦИЯ)
# -------------------------------------------------
def recommend_for_user(target_user, user_ratings, all_users, movie_id_to_title, n_rec=5, k_neighbors=10):
    """
    Рекомендует фильмы пользователю на основе похожих пользователей
    """
    # 1. Найти похожих пользователей
    similar_users = get_top_similar_users(target_user, user_ratings, all_users, top_k=k_neighbors)
    
    if not similar_users:
        return [], 0.0  # Нет похожих пользователей
    
    # 2. Собрать все фильмы, которые оценили похожие пользователи
    candidate_movies = defaultdict(float)  # movie_id -> суммарное взвешенное сходство
    
    for similar_user, sim_score in similar_users:
        # Фильмы, которые оценил похожий пользователь
        for movie_id, rating in user_ratings[similar_user].items():
            # Пропускаем, если пользователь уже оценил этот фильм
            if movie_id in user_ratings.get(target_user, {}):
                continue
            # Взвешиваем по сходству
            candidate_movies[movie_id] += sim_score * rating
    
    # 3. Сортируем по взвешенному рейтингу
    sorted_candidates = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Берём топ-N
    recommendations = [mid for mid, score in sorted_candidates[:n_rec]]
    
    # 5. Среднее сходство похожих пользователей (для информации)
    avg_similarity = sum(sim for _, sim in similar_users) / len(similar_users)
    
    return recommendations, avg_similarity

# -------------------------------------------------
# 6. МЕТРИКА КАЧЕСТВА: HIT RATE @K
# -------------------------------------------------
def hit_rate_at_k(user_ratings_train, user_ratings_test, all_users_train, movie_id_to_title, k_neighbors=10, n_rec=5):
    """
    Hit Rate @K: доля пользователей, для которых хотя бы один рекомендованный фильм
    совпадает с тем, что они оценили в test
    """
    hits = 0
    total = 0
    
    # Для каждого пользователя, который есть и в train, и в test
    for user in user_ratings_test.keys():
        if user not in user_ratings_train:
            continue
        
        # Получаем рекомендации на основе train
        recs, _ = recommend_for_user(
            user, user_ratings_train, all_users_train,
            movie_id_to_title, n_rec=n_rec, k_neighbors=k_neighbors
        )
        
        if not recs:
            continue
        
        # Фильмы, которые пользователь оценил в test
        test_movies = set(user_ratings_test[user].keys())
        
        # Проверяем пересечение
        if set(recs) & test_movies:
            hits += 1
        total += 1
    
    return hits / total if total > 0 else 0.0

# -------------------------------------------------
# 7. ОСНОВНАЯ ПРОГРАММА
# -------------------------------------------------
if __name__ == "__main__":
    # Параметры
    RATING_FILE = "ml-1m/ratings.dat"
    MOVIE_FILE = "ml-1m/movies.dat"
    TEST_RATIO = 0.2
    K_NEIGHBORS = 10  # Число похожих пользователей
    N_REC = 5         # Сколько фильмов рекомендовать
    
    print("=" * 60)
    print("USER-BASED РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")
    print("=" * 60)
    
    # Загрузка данных
    print("\n📥 Загрузка данных...")
    train_df, test_df, id2title = load_data(RATING_FILE, MOVIE_FILE, test_ratio=TEST_RATIO)
    
    # Построение моделей
    print("🔧 Построение моделей...")
    user_ratings_train = build_user_ratings(train_df)
    user_ratings_test = build_user_ratings(test_df)
    all_users_train = list(user_ratings_train.keys())
    
    # Оценка качества
    print("📊 Оценка качества (Hit Rate @K)...")
    hr = hit_rate_at_k(
        user_ratings_train, user_ratings_test, all_users_train, id2title,
        k_neighbors=K_NEIGHBORS, n_rec=N_REC
    )
    print(f"   Hit Rate @{N_REC}: {hr:.4f}")
    print(f"   Это означает: в {(hr*100):.1f}% случаев хотя бы один рекомендованный фильм")
    print(f"   пользователь действительно оценил в тестовых данных.\n")
    
    # Интерактивный режим
    print("=" * 60)
    print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 60)
    print(f"Примеры популярных пользователей: {random.sample(all_users_train[:100], 5)}")
    print()
    
    while True:
        try:
            user_input = input("Введите ID пользователя (или 'quit' для выхода): ").strip()
            if user_input.lower() == 'quit':
                break
            
            user_id = int(user_input)
            
            if user_id not in user_ratings_train:
                print(f"⚠️  Пользователь {user_id} не найден в обучающих данных.")
                print(f"   Попробуйте одного из: {random.sample(all_users_train[:100], 5)}")
                continue
            
            # Получаем рекомендации
            recs, avg_sim = recommend_for_user(
                user_id, user_ratings_train, all_users_train, id2title,
                n_rec=N_REC, k_neighbors=K_NEIGHBORS
            )
            
            # Выводим информацию о пользователе
            print(f"\n👤 Пользователь {user_id}")
            print(f"   Оценил фильмов: {len(user_ratings_train[user_id])}")
            print(f"   Среднее сходство с соседями: {avg_sim:.3f}")
            
            if not recs:
                print("   ❌ Нет рекомендаций (нет похожих пользователей или все фильмы уже оценены)")
                print()
                continue
            
            # Выводим рекомендации
            print(f"\n🎬 Рекомендации (топ-{N_REC}):")
            for i, mid in enumerate(recs, 1):
                title = id2title.get(mid, f"[ID {mid}]")
                print(f"   {i}. {title}")
            
            # Показываем, что пользователь уже смотрел
            print(f"\n📋 Несколько фильмов, которые пользователь уже оценил:")
            user_movies = list(user_ratings_train[user_id].keys())[:5]
            for mid in user_movies:
                title = id2title.get(mid, f"[ID {mid}]")
                rating = user_ratings_train[user_id][mid]
                print(f"   • {title} (оценка: {rating})")
            
            print()
        
        except ValueError:
            print("❌ Пожалуйста, введите целое число.")
        except KeyboardInterrupt:
            break
    
    print("\n👋 До свидания!")