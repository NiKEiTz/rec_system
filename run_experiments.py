import json
import csv
import os
from datetime import datetime
from tqdm import tqdm

# Импорт функций из основного модуля
from recommender_hybrid import (
    load_data,
    build_sparse_matrices,
    build_similarity_matrices,
    hybrid_recommendations_optimized,
    hit_rate_at_k_optimized,
    precision_at_k_optimized,
    recall_at_k_optimized
)

def load_config(config_file='experiments_config.json'):
    """Загрузка конфигурации экспериментов"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_data():
    """Подготовка данных (один раз для всех экспериментов)"""
    print("📥 Загрузка данных...")
    train_df, test_df, _ = load_data(
        "data/ratings.dat", 
        "data/movies.dat"
    )
    
    all_users = sorted(pd.concat([train_df['user_id'], test_df['user_id']]).unique())
    all_items = sorted(pd.concat([train_df['movie_id'], test_df['movie_id']]).unique())
    
    user_item_train, item_user_train, user_idx_train, item_idx_train = build_sparse_matrices(
        train_df, all_users, all_items
    )
    user_item_test, item_user_test, _, _ = build_sparse_matrices(
        test_df, all_users, all_items
    )
    
    print("⚡ Построение матриц сходства...")
    user_sim_topk, item_sim_topk = build_similarity_matrices(
        user_item_train, item_user_train, top_k=50
    )
    
    return (user_item_train, user_item_test, user_sim_topk, item_sim_topk, 
            user_idx_train, item_idx_train)

def run_single_experiment(params, matrices):
    """Запуск одного эксперимента"""
    (user_item_train, user_item_test, user_sim_topk, 
     item_sim_topk, user_idx_train, item_idx_train) = matrices
    
    # Убираем служебные поля
    exp_params = {k: v for k, v in params.items() 
                  if k not in ['name', 'type', 'description']}
    
    # Запуск метрик
    hr = hit_rate_at_k_optimized(
        hybrid_recommendations_optimized,
        user_item_train, user_item_test,
        user_sim_topk, item_sim_topk,
        user_idx_train, item_idx_train,
        max_test_users=200,
        **exp_params
    )
    
    prec = precision_at_k_optimized(
        hybrid_recommendations_optimized,
        user_item_train, user_item_test,
        user_sim_topk, item_sim_topk,
        max_test_users=200,
        **exp_params
    )
    
    rec = recall_at_k_optimized(
        hybrid_recommendations_optimized,
        user_item_train, user_item_test,
        user_sim_topk, item_sim_topk,
        max_test_users=200,
        **exp_params
    )
    
    # Формирование результата
    return {
        'experiment_name': params['name'],
        'type': params['type'],
        'description': params['description'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hit_rate@10': round(hr, 4),
        'precision@10': round(prec, 4),
        'recall@10': round(rec, 4),
        **exp_params
    }

def save_results_to_csv(results, filename='experiments_results.csv'):
    """Сохранение результатов в CSV"""
    if not results:
        return
    
    # Заголовки из первого результата
    fieldnames = list(results[0].keys())
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)
    
    print(f"\n✅ Результаты сохранены в: {filename}")

def main():
    print("=" * 70)
    print("ЗАПУСК ТЕСТОВ С ГИПЕРПАРАМЕТРАМИ")
    print("=" * 70)
    
    # Загрузка конфигурации
    config = load_config()
    tests = config['tests']
    print(f"📋 Найдено {len(tests)} тестов для запуска\n")
    
    # Подготовка данных (один раз)
    matrices = prepare_data()
    
    # Запуск всех тестов
    results = []
    print("\n🧪 Запуск тестов:\n")
    
    for i, test_params in enumerate(tqdm(tests, desc="Прогресс"), 1):
        tqdm.write(f"\n[{i}/{len(tests)}] Тест: {test_params['name']} ({test_params['type']})")
        result = run_single_experiment(test_params, matrices)
        results.append(result)
        tqdm.write(f"   ✅ Hit Rate @10: {result['hit_rate@10']:.4f}")
    
    # Сохранение результатов
    save_results_to_csv(results)
    
    # Вывод сводки
    print("\n" + "=" * 70)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 70)
    
    # Группировка по типу теста
    boost_tests = [r for r in results if r['type'] == 'boost_factor']
    k_tests = [r for r in results if r['type'] == 'k_neighbors']
    
    if boost_tests:
        print("\n📈 Влияние boost_factor на Hit Rate @10:")
        print("   " + "-" * 50)
        for r in sorted(boost_tests, key=lambda x: x['boost_factor']):
            print(f"   boost_factor={r['boost_factor']:4.1f} → Hit Rate: {r['hit_rate@10']:.4f} | Precision: {r['precision@10']:.4f}")
    
    if k_tests:
        print("\n📈 Влияние k_neighbors на Hit Rate @10:")
        print("   " + "-" * 50)
        for r in sorted(k_tests, key=lambda x: x['k_neighbors']):
            print(f"   k_neighbors={r['k_neighbors']:2d} → Hit Rate: {r['hit_rate@10']:.4f} | Precision: {r['precision@10']:.4f}")
    
    print("\n" + "=" * 70)
    print("Готово! Для визуализации запустите: python visualize_results.py")
    print("=" * 70)

if __name__ == "__main__":
    import pandas as pd
    main()