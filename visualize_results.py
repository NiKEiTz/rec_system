import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_results(filename='experiments_results.csv'):
    """Загрузка результатов из CSV"""
    if not os.path.exists(filename):
        print(f"❌ Файл {filename} не найден. Сначала запустите: python run_experiments.py")
        return None
    
    return pd.read_csv(filename)

def plot_boost_factor_analysis(df, save_path='boost_factor_analysis.png'):
    """График влияния boost_factor"""
    boost_df = df[df['type'] == 'boost_factor'].copy()
    
    if boost_df.empty:
        print("⚠️ Нет данных для анализа boost_factor")
        return
    
    boost_df = boost_df.sort_values('boost_factor')
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Основной график
    plt.plot(boost_df['boost_factor'], boost_df['hit_rate@10'], 
             marker='o', linewidth=2, markersize=8, label='Hit Rate @10', color='#2E86AB')
    plt.plot(boost_df['boost_factor'], boost_df['precision@10'], 
             marker='s', linewidth=2, markersize=8, label='Precision @10', color='#A23B72')
    plt.plot(boost_df['boost_factor'], boost_df['recall@10'], 
             marker='^', linewidth=2, markersize=8, label='Recall @10', color='#F18F01')
    
    # Оптимальная точка
    best_idx = boost_df['hit_rate@10'].idxmax()
    plt.axvline(x=boost_df.loc[best_idx, 'boost_factor'], 
                color='green', linestyle='--', alpha=0.7, label='Оптимум')
    
    plt.xlabel('boost_factor', fontsize=12, fontweight='bold')
    plt.ylabel('Значение метрики', fontsize=12, fontweight='bold')
    plt.title('Влияние boost_factor на качество рекомендаций', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранён: {save_path}")
    plt.close()

def plot_k_neighbors_analysis(df, save_path='k_neighbors_analysis.png'):
    """График влияния k_neighbors"""
    k_df = df[df['type'] == 'k_neighbors'].copy()
    
    if k_df.empty:
        print("⚠️ Нет данных для анализа k_neighbors")
        return
    
    k_df = k_df.sort_values('k_neighbors')
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    plt.plot(k_df['k_neighbors'], k_df['hit_rate@10'], 
             marker='o', linewidth=2, markersize=8, label='Hit Rate @10', color='#2E86AB')
    plt.plot(k_df['k_neighbors'], k_df['precision@10'], 
             marker='s', linewidth=2, markersize=8, label='Precision @10', color='#A23B72')
    plt.plot(k_df['k_neighbors'], k_df['recall@10'], 
             marker='^', linewidth=2, markersize=8, label='Recall @10', color='#F18F01')
    
    # Оптимальная точка
    best_idx = k_df['hit_rate@10'].idxmax()
    plt.axvline(x=k_df.loc[best_idx, 'k_neighbors'], 
                color='green', linestyle='--', alpha=0.7, label='Оптимум')
    
    plt.xlabel('k_neighbors', fontsize=12, fontweight='bold')
    plt.ylabel('Значение метрики', fontsize=12, fontweight='bold')
    plt.title('Влияние k_neighbors на качество рекомендаций', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранён: {save_path}")
    plt.close()

def print_summary_table(df):
    """Печать сводной таблицы"""
    print("\n" + "=" * 100)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ С ГИПЕРПАРАМЕТРАМИ")
    print("=" * 100)
    
    # Тесты boost_factor
    boost_df = df[df['type'] == 'boost_factor'].copy()
    if not boost_df.empty:
        boost_df = boost_df.sort_values('boost_factor')
        print("\n📊 Тесты: влияние boost_factor (фиксированные остальные параметры)")
        print("-" * 100)
        print(f"{'boost_factor':<15} {'Hit Rate@10':<15} {'Precision@10':<15} {'Recall@10':<15} {'Описание':<30}")
        print("-" * 100)
        for _, row in boost_df.iterrows():
            marker = " ← ОПТИМУМ" if row['hit_rate@10'] == boost_df['hit_rate@10'].max() else ""
            print(f"{row['boost_factor']:<15.1f} {row['hit_rate@10']:<15.4f} {row['precision@10']:<15.4f} "
                  f"{row['recall@10']:<15.4f} {row['description']:<30}{marker}")
    
    # Тесты k_neighbors
    k_df = df[df['type'] == 'k_neighbors'].copy()
    if not k_df.empty:
        k_df = k_df.sort_values('k_neighbors')
        print("\n📊 Тесты: влияние k_neighbors (фиксированные остальные параметры)")
        print("-" * 100)
        print(f"{'k_neighbors':<15} {'Hit Rate@10':<15} {'Precision@10':<15} {'Recall@10':<15} {'Описание':<30}")
        print("-" * 100)
        for _, row in k_df.iterrows():
            marker = " ← ОПТИМУМ" if row['hit_rate@10'] == k_df['hit_rate@10'].max() else ""
            print(f"{row['k_neighbors']:<15d} {row['hit_rate@10']:<15.4f} {row['precision@10']:<15.4f} "
                  f"{row['recall@10']:<15.4f} {row['description']:<30}{marker}")
    
    print("=" * 100)

def plot_relative_improvement(df):
    """График относительного улучшения метрик"""
    boost_df = df[df['type'] == 'boost_factor'].sort_values('boost_factor')
    k_df = df[df['type'] == 'k_neighbors'].sort_values('k_neighbors')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boost factor
    baseline_hr = boost_df.iloc[0]['hit_rate@10']
    boost_df['relative_improvement'] = ((boost_df['hit_rate@10'] - baseline_hr) / baseline_hr) * 100
    
    ax1.plot(boost_df['boost_factor'], boost_df['relative_improvement'], 
             marker='o', linewidth=2, color='#2E86AB')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('boost_factor', fontweight='bold')
    ax1.set_ylabel('Относительный прирост Hit Rate, %', fontweight='bold')
    ax1.set_title('Влияние boost_factor (база = 1.0)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # K neighbors
    baseline_k = k_df.iloc[0]['hit_rate@10']
    k_df['relative_improvement'] = ((k_df['hit_rate@10'] - baseline_k) / baseline_k) * 100
    
    ax2.plot(k_df['k_neighbors'], k_df['relative_improvement'], 
             marker='o', linewidth=2, color='#A23B72')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('k_neighbors', fontweight='bold')
    ax2.set_ylabel('Относительный прирост Hit Rate, %', fontweight='bold')
    ax2.set_title('Влияние k_neighbors (база = 5)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('relative_improvement.png', dpi=300, bbox_inches='tight')
    print("✅ График относительного улучшения: relative_improvement.png")

def main():
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ТЕСТОВ")
    print("=" * 70)
    
    df = load_results()
    if df is None:
        return
    
    # Печать сводной таблицы
    print_summary_table(df)
    
    # Построение графиков
    print("\n📈 Построение графиков...")
    plot_boost_factor_analysis(df)
    plot_k_neighbors_analysis(df)
    plot_relative_improvement(df)
    
    print("\n" + "=" * 70)
    print("Готово! Графики сохранены в текущую директорию.")
    print("=" * 70)

if __name__ == "__main__":
    main()