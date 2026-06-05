"""
MIE1628 Assignment 2 - Part B-2
Split Dataset and Performance Assessment
"""

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("MovieRecommender-PartB2").getOrCreate()

# 1. Load Data
print("\n# 1. Load Dataset")
df = spark.read.csv("movies.csv",
                    header=True, inferSchema=True)
print(f"Total records: {df.count():,}")

# 2. Configure Split Ratios and Evaluator
print("\n# 2. Configure Split Ratios")
split_ratios = [
    (0.70, 0.30, "70/30"),
    (0.80, 0.20, "80/20")
]

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)


# 3. Configure ALS Model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
              rank=10, maxIter=10, regParam=0.1,
              coldStartStrategy="drop")

## 4. Train and Evaluate Models
results = []

for train_pct, test_pct, label in split_ratios:
    print(f"\n--- Evaluating {label} split ---")

    # Split dataset
    train_df, test_df = df.randomSplit([train_pct, test_pct], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"  Train size: {train_count:,} | Test size: {test_count:,}")



    model = als.fit(train_df)

    # Predict and evaluate
    predictions = model.transform(test_df)
    rmse = evaluator.evaluate(predictions)

    print(f"RMSE: {rmse:.4f}")

    # Save results
    results.append({
        'split_ratio': label,
        'train_count': train_count,
        'test_count': test_count,
        'rmse': rmse
    })


# 5. Display Results
print(f"\n{'Split Ratio':<15} {'Train Size':<12} {'Test Size':<12} {'RMSE':<10}")
print("-" * 50)
for r in results:
    print(f"{r['split_ratio']:<15} {r['train_count']:<12,} {r['test_count']:<12,} {r['rmse']:<10.4f}")

# Identify best/worst
best_split = min(results, key=lambda x: x['rmse'])
worst_split = max(results, key=lambda x: x['rmse'])

print(f"\nBest performing split:  {best_split['split_ratio']} (RMSE: {best_split['rmse']:.4f})")
print(f"Worst performing split: {worst_split['split_ratio']} (RMSE: {worst_split['rmse']:.4f})")
print(f"Performance difference: {abs(worst_split['rmse'] - best_split['rmse']):.4f}")


# 6. Visualization (Performance Comparison)
import matplotlib.pyplot as plt

rmse_diff = abs(results[0]['rmse'] - results[1]['rmse'])
try:
    labels = [r['split_ratio'] for r in results]
    rmse_values = [r['rmse'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, rmse_values, color=['steelblue', 'coral'],
                  alpha=0.8, edgecolor='black', width=0.5)

    ax.set_xlabel('Train/Test Split Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: 70/30 vs 80/20 Split', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, rmse in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rmse:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add performance difference annotation
    mid_x = 0.5
    mid_y = (rmse_values[0] + rmse_values[1]) / 2
    ax.annotate('', xy=(0, rmse_values[0]), xytext=(1, rmse_values[1]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(mid_x, mid_y, f'Δ = {rmse_diff:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('split_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Visualization error: {e}")

spark.stop()

print("\n----- Complete -----\n")
