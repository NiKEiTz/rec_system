import pandas as pd
import numpy as np
from collections import defaultdict
import random
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Прогресс-бар

# ════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА ДАННЫХ (Leave-one-out)
# ════════════════════════════════════════════════════════════════════

def load_data(ratings_path: str, movies_path: str, seed: int = 42): 
    """
    Leave-One-Out: для каждого пользователя скрываем 1 случайную оценку в тест
    """
    print("📥 Загрузка данных...")
    ratings = pd.read_csv(
        ratings_path,
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    movies = pd.read_csv(
        movies_path,
        sep='::',
        names=['movie_id', 'title', 'genres'],
        encoding='latin1',
        engine='python'
    )
    movie_id_to_title = dict(zip(movies['movie_id'], movies['title']))
    
    # Для каждого пользователя выбираем 1 случайную оценку для теста
    np.random.seed(seed)
    test_ratings_list = []
    
    for user_id, group in ratings.groupby('user_id'):
        if len(group) > 1:  # У пользователя должно быть >1 оценки
            test_row = group.sample(n=1)
            test_ratings_list.append(test_row)
    
    test_ratings = pd.concat(test_ratings_list)
    train_ratings = ratings.drop(test_ratings.index)
    
    print(f"   Всего пользователей: {ratings['user_id'].nunique()}")
    print(f"   Train: {len(train_ratings)} оценок")
    print(f"   Test:  {len(test_ratings)} оценок (по 1 на пользователя)")
    
    return train_ratings, test_ratings, movie_id_to_title

# ════════════════════════════════════════════════════════════════════
# 2. ПОСТРОЕНИЕ РАЗРЕЖЕННЫХ МАТРИЦ
# ════════════════════════════════════════════════════════════════════

def build_sparse_matrices(df, all_users=None, all_items=None):
    """
    Построение разреженных матриц:
    - user_item_matrix: пользователи × фильмы
    - item_user_matrix: фильмы × пользователи
    """
    print("🔧 Построение разреженных матриц...")
    
    if all_users is None:
        all_users = sorted(df['user_id'].unique())
    if all_items is None:
        all_items = sorted(df['movie_id'].unique())
    
    user_idx = {u: i for i, u in enumerate(all_users)}
    item_idx = {i: j for j, i in enumerate(all_items)}
    
    # Матрица пользователь-фильм
    rows_ui, cols_ui, data_ui = [], [], []
    # Матрица фильм-пользователь
    rows_iu, cols_iu, data_iu = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Обработка оценок"):
        u_idx = user_idx[row['user_id']]
        i_idx = item_idx[row['movie_id']]
        rating = row['rating']
        
        rows_ui.append(u_idx)
        cols_ui.append(i_idx)
        data_ui.append(rating)
        
        rows_iu.append(i_idx)
        cols_iu.append(u_idx)
        data_iu.append(rating)
    
    user_item_matrix = csr_matrix((data_ui, (rows_ui, cols_ui)), shape=(len(all_users), len(all_items)))
    item_user_matrix = csr_matrix((data_iu, (rows_iu, cols_iu)), shape=(len(all_items), len(all_users)))
    
    print(f"   Матрица пользователь-фильм: {user_item_matrix.shape}, ненулевых: {user_item_matrix.nnz}")
    print(f"   Матрица фильм-пользователь: {item_user_matrix.shape}, ненулевых: {item_user_matrix.nnz}")
    
    return user_item_matrix, item_user_matrix, user_idx, item_idx

# ════════════════════════════════════════════════════════════════════
# 3. ПОСТРОЕНИЕ МАТРИЦ СХОДСТВА
# ════════════════════════════════════════════════════════════════════

def build_similarity_matrices(user_item_matrix, item_user_matrix, top_k=50):
    """
    Предвычисление матриц сходства с использованием косинусного сходства
    Возвращает топ-K похожих для каждого пользователя/фильма
    """
    print("⚡ Расчёт матриц сходства...")
    
    # Сходство между пользователями
    print("   Сходство пользователей...")
    user_similarity = cosine_similarity(user_item_matrix)
    np.fill_diagonal(user_similarity, 0)  # Сам с собой = 0
    
    # Сходство между фильмами
    print("   Сходство фильмов...")
    item_similarity = cosine_similarity(item_user_matrix)
    np.fill_diagonal(item_similarity, 0)  # Сам с собой = 0
    
    # Преобразуем в формат: {объект: [(похожий_объект, сходство), ...]}
    print("   Формирование топ-K списков...")
    
    user_sim_topk = {}
    for i in tqdm(range(len(user_similarity)), desc="   Пользователи"):
        sims = user_similarity[i]
        top_indices = np.argsort(sims)[::-1][:top_k]
        top_sims = sims[top_indices]
        user_sim_topk[i] = [(idx, sim) for idx, sim in zip(top_indices, top_sims) if sim > 0]
    
    item_sim_topk = {}
    for i in tqdm(range(len(item_similarity)), desc="   Фильмы"):
        sims = item_similarity[i]
        top_indices = np.argsort(sims)[::-1][:top_k]
        top_sims = sims[top_indices]
        item_sim_topk[i] = [(idx, sim) for idx, sim in zip(top_indices, top_sims) if sim > 0]
    
    print(f"   Готово! Кэшировано сходство для {len(user_sim_topk)} пользователей и {len(item_sim_topk)} фильмов")
    
    return user_sim_topk, item_sim_topk

# ════════════════════════════════════════════════════════════════════
# 4. USER-BASED РЕКОМЕНДАЦИИ (ОПТИМИЗИРОВАННЫЕ)
# ════════════════════════════════════════════════════════════════════

def user_based_recommendations_optimized(user_idx: int, user_item_matrix, user_sim_topk, 
                                          k_neighbors: int = 10, n_candidates: int = 50):
    """
    User-based рекомендации с использованием предвычисленного сходства
    Возвращает: {фильм_idx: score}
    """
    similar_users = user_sim_topk.get(user_idx, [])[:k_neighbors]
    
    if not similar_users:
        return {}
    
    candidate_scores = defaultdict(float)
    user_vector = user_item_matrix[user_idx].toarray()[0]
    watched_mask = user_vector > 0
    watched_indices = set(np.where(watched_mask)[0])
    
    for sim_user_idx, sim_score in similar_users:
        sim_user_vector = user_item_matrix[sim_user_idx].toarray()[0]
        
        # Находим фильмы, которые оценил похожий пользователь, но не текущий
        for movie_idx in np.where(sim_user_vector > 0)[0]:
            if movie_idx in watched_indices:
                continue
            rating = sim_user_vector[movie_idx]
            candidate_scores[movie_idx] += sim_score * rating
    
    # Сортируем по убыванию
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_candidates[:n_candidates])

# ════════════════════════════════════════════════════════════════════
# 5. ITEM-BASED РЕКОМЕНДАЦИИ (ОПТИМИЗИРОВАННЫЕ)
# ════════════════════════════════════════════════════════════════════

def item_based_recommendations_optimized(user_idx: int, user_item_matrix, item_sim_topk,
                                          top_k_per_item: int = 5, n_candidates: int = 50):
    """
    Item-based рекомендации с использованием предвычисленного сходства
    Возвращает: {фильм_idx: score}
    """
    user_vector = user_item_matrix[user_idx].toarray()[0]
    user_rated_indices = np.where(user_vector > 0)[0]
    
    if len(user_rated_indices) == 0:
        return {}
    
    candidate_scores = defaultdict(float)
    watched_set = set(user_rated_indices)
    
    for movie_idx in user_rated_indices:
        rating = user_vector[movie_idx]
        similar_items = item_sim_topk.get(movie_idx, [])[:top_k_per_item]
        
        for sim_item_idx, sim_score in similar_items:
            if sim_item_idx in watched_set:
                continue
            candidate_scores[sim_item_idx] += sim_score * rating
    
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_candidates[:n_candidates])

# ════════════════════════════════════════════════════════════════════
# 6. ГИБРИДНАЯ СИСТЕМА
# ════════════════════════════════════════════════════════════════════

def hybrid_recommendations_optimized(user_idx: int, user_item_matrix, user_sim_topk, item_sim_topk,
                                     alpha: float = 0.5, boost_factor: float = 1.5,
                                     k_neighbors: int = 10, top_k_per_item: int = 5,
                                     n_candidates: int = 50, n_final: int = 10):
    """
    Гибридная рекомендательная система с оптимизированными расчётами
    Возвращает: [(фильм_idx, score, source), ...]
    """
    # Получаем рекомендации от обоих подходов
    user_based = user_based_recommendations_optimized(
        user_idx, user_item_matrix, user_sim_topk,
        k_neighbors=k_neighbors, n_candidates=n_candidates
    )
    
    item_based = item_based_recommendations_optimized(
        user_idx, user_item_matrix, item_sim_topk,
        top_k_per_item=top_k_per_item, n_candidates=n_candidates
    )
    
    # Нормализуем оценки
    def normalize_scores(scores):
        if not scores:
            return {}
        values = np.array(list(scores.values()))
        min_v, max_v = values.min(), values.max()
        if max_v == min_v:
            return {k: 1.0 for k in scores.keys()}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
    
    user_norm = normalize_scores(user_based)
    item_norm = normalize_scores(item_based)
    
    # Объединяем с усилением пересечений
    all_candidates = set(user_norm.keys()) | set(item_norm.keys())
    final_scores = {}
    
    for movie_idx in all_candidates:
        score_user = user_norm.get(movie_idx, 0.0)
        score_item = item_norm.get(movie_idx, 0.0)
        
        if score_user > 0 and score_item > 0:
            combined = alpha * score_user + (1 - alpha) * score_item
            combined *= boost_factor
            source = "BOTH"
        elif score_user > 0:
            combined = score_user
            source = "USER"
        else:
            combined = score_item
            source = "ITEM"
        
        final_scores[movie_idx] = (combined, source)
    
    # Сортируем и возвращаем топ-N
    sorted_final = sorted(final_scores.items(), key=lambda x: x[1][0], reverse=True)
    return [(mid, score, src) for mid, (score, src) in sorted_final[:n_final]]

# ════════════════════════════════════════════════════════════════════
# 7. МЕТРИКИ КАЧЕСТВА (С ПРОГРЕСС-БАРОМ)
# ════════════════════════════════════════════════════════════════════

def hit_rate_at_k_optimized(hybrid_func, user_item_matrix_train, user_item_matrix_test,
                            user_sim_topk, item_sim_topk, user_idx_map, item_idx_map,
                            max_test_users=500, **kwargs):
    """Hit Rate @K с прогресс-баром"""
    hits = 0
    total = 0
    
    # Берём только часть пользователей для быстрой оценки
    test_user_indices = [i for i in range(user_item_matrix_test.shape[0]) 
                        if user_item_matrix_test[i].nnz > 0][:max_test_users]
    
    print(f"\n📊 Расчёт метрик для {len(test_user_indices)} пользователей...")
    
    for user_idx in tqdm(test_user_indices, desc="   Hit Rate"):
        # Получаем рекомендации
        recs = hybrid_func(
            user_idx, user_item_matrix_train, user_sim_topk, item_sim_topk, **kwargs
        )
        
        if not recs:
            continue
        
        rec_indices = set(mid for mid, _, _ in recs)
        test_vector = user_item_matrix_test[user_idx].toarray()[0]
        test_movies = set(np.where(test_vector > 0)[0])
        
        if rec_indices & test_movies:
            hits += 1
        total += 1
    
    return hits / total if total > 0 else 0.0

def precision_at_k_optimized(hybrid_func, user_item_matrix_train, user_item_matrix_test,
                             user_sim_topk, item_sim_topk, max_test_users=500, **kwargs):
    """Precision @K с прогресс-баром"""
    total_precision = 0
    count = 0
    
    test_user_indices = [i for i in range(user_item_matrix_test.shape[0]) 
                        if user_item_matrix_test[i].nnz > 0][:max_test_users]
    
    for user_idx in tqdm(test_user_indices, desc="   Precision"):
        recs = hybrid_func(
            user_idx, user_item_matrix_train, user_sim_topk, item_sim_topk, **kwargs
        )
        
        if not recs:
            continue
        
        rec_indices = set(mid for mid, _, _ in recs)
        test_vector = user_item_matrix_test[user_idx].toarray()[0]
        test_movies = set(np.where(test_vector > 0)[0])
        
        precision = len(rec_indices & test_movies) / len(rec_indices) if rec_indices else 0
        total_precision += precision
        count += 1
    
    return total_precision / count if count > 0 else 0.0

def recall_at_k_optimized(hybrid_func, user_item_matrix_train, user_item_matrix_test,
                          user_sim_topk, item_sim_topk, max_test_users=500, **kwargs):
    """Recall @K с прогресс-баром"""
    total_recall = 0
    count = 0
    
    test_user_indices = [i for i in range(user_item_matrix_test.shape[0]) 
                        if user_item_matrix_test[i].nnz > 0][:max_test_users]
    
    for user_idx in tqdm(test_user_indices, desc="   Recall"):
        recs = hybrid_func(
            user_idx, user_item_matrix_train, user_sim_topk, item_sim_topk, **kwargs
        )
        
        if not recs:
            continue
        
        rec_indices = set(mid for mid, _, _ in recs)
        test_vector = user_item_matrix_test[user_idx].toarray()[0]
        test_movies = set(np.where(test_vector > 0)[0])
        
        recall = len(rec_indices & test_movies) / len(test_movies) if test_movies else 0
        total_recall += recall
        count += 1
    
    return total_recall / count if count > 0 else 0.0

# ════════════════════════════════════════════════════════════════════
# 8. ОСНОВНАЯ ПРОГРАММА
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Параметры
    RATING_FILE = "data/ratings.dat"
    MOVIE_FILE = "data/movies.dat"
    
    HYPERPARAMS = {
        'alpha': 0.5,
        'boost_factor': 1.5,        # Важность пересечения
        'k_neighbors': 30,          # ub 30 похожих пользователей
        'top_k_per_item': 10,       # ib 10 фильмов, похожих на каждый
        'n_candidates': 50,         # 50 фильмов, которые попали в топ
        'n_final': 25               # Итог по пользователю
    }
    
    print("=" * 70)
    print("ГИБРИДНАЯ РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА (с оптимизацией через scipy)")
    print("=" * 70)
    
    # Загрузка данных
    train_df, test_df, id2title = load_data(RATING_FILE, MOVIE_FILE)
    
    # Построение матриц
    all_users = sorted(pd.concat([train_df['user_id'], test_df['user_id']]).unique())
    all_items = sorted(pd.concat([train_df['movie_id'], test_df['movie_id']]).unique())
    
    user_item_train, item_user_train, user_idx_train, item_idx_train = build_sparse_matrices(
        train_df, all_users, all_items
    )
    user_item_test, item_user_test, _, _ = build_sparse_matrices(
        test_df, all_users, all_items
    )
    
    # Построение матриц сходства
    user_sim_topk, item_sim_topk = build_similarity_matrices(
        user_item_train, item_user_train, top_k=50
    )
    
    # Оценка качества
    print("\n📊 Оценка качества (метрики)...")
    
    hr = hit_rate_at_k_optimized(
        hybrid_recommendations_optimized,
        user_item_train, user_item_test,
        user_sim_topk, item_sim_topk,
        user_idx_train, item_idx_train,
        max_test_users=500,
        **HYPERPARAMS
    )
    
    prec = precision_at_k_optimized(
        hybrid_recommendations_optimized,
        user_item_train, user_item_test,
        user_sim_topk, item_sim_topk,
        max_test_users=500,
        **HYPERPARAMS
    )
    
    rec = recall_at_k_optimized(
        hybrid_recommendations_optimized,
        user_item_train, user_item_test,
        user_sim_topk, item_sim_topk,
        max_test_users=500,
        **HYPERPARAMS
    )
    
    print(f"\n✅ Результаты:")
    print(f"   Hit Rate @10:    {hr:.4f}")
    print(f"   Precision @10:   {prec:.4f}")
    print(f"   Recall @10:      {rec:.4f}")
    
    # Интерактивный режим
    print("\n" + "=" * 70)
    print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 70)
    
    # Инвертируем словари для обратного поиска
    idx_to_user = {idx: user_id for user_id, idx in user_idx_train.items()}
    idx_to_item = {idx: item_id for item_id, idx in item_idx_train.items()}
    
    print(f"Примеры пользователей: {list(idx_to_user.values())[:5]}\n")
    
    while True:
        try:
            user_input = input("Введите ID пользователя (или 'quit' для выхода): ").strip()
            if user_input.lower() == 'quit':
                break
            
            user_id = int(user_input)
            
            if user_id not in user_idx_train:
                print(f"⚠️  Пользователь {user_id} не найден.")
                print(f"   Попробуйте: {list(idx_to_user.values())[:5]}")
                continue
            
            user_idx = user_idx_train[user_id]
            
            # Получаем рекомендации
            recs = hybrid_recommendations_optimized(
                user_idx, user_item_train, user_sim_topk, item_sim_topk,
                **HYPERPARAMS
            )
            
            # Информация о пользователе
            user_vector = user_item_train[user_idx].toarray()[0]
            n_watched = np.count_nonzero(user_vector)
            
            print(f"\n{'='*60}")
            print(f"👤 Пользователь {user_id} (индекс: {user_idx})")
            print(f"   Оценил фильмов: {n_watched}")
            print(f"\n🎬 Рекомендации (гибридная система):")
            
            # Вывод рекомендаций
            for i, (movie_idx, score, source) in enumerate(recs, 1):
                movie_id = idx_to_item.get(movie_idx, movie_idx)
                title = id2title.get(movie_id, f"[ID {movie_id}]")
                source_icon = {"BOTH": "⭐", "USER": "👥", "ITEM": "🎬"}
                print(f"   {i}. {source_icon[source]} {title}")
                print(f"      Score: {score:.3f} | Источник: {source}")
            
            # Показываем, что пользователь уже смотрел
            print(f"\n📋 Несколько фильмов, которые пользователь уже оценил:")
            watched_indices = np.where(user_vector > 0)[0][:5]
            for movie_idx in watched_indices:
                movie_id = idx_to_item.get(movie_idx, movie_idx)
                title = id2title.get(movie_id, f"[ID {movie_id}]")
                rating = user_vector[movie_idx]
                print(f"   • {title} (оценка: {rating})")
            
            print(f"\n{'='*60}\n")
        
        except ValueError:
            print("❌ Пожалуйста, введите целое число.")
        except KeyboardInterrupt:
            break
    
    print("\n👋 До свидания!")