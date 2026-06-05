"""
MIE1628 Assignment 2 - Part B-4
Hyperparameter Tuning Using Cross-Validation Techniques

"""

import os
import sys
import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize Spark
spark = SparkSession.builder.appName("MovieRecommender-PartB4").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 1. Load and Prepare Data

df = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load("movies.csv")
    .select("userId", "movieId", "rating")
)
print(f"Total records: {df.count():,}")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count():,} | Test: {test_df.count():,}")

# 2. ALS setup

als = ALS(
    userCol='userId',
    itemCol='movieId',
    ratingCol='rating',
    coldStartStrategy='drop',
    seed=42
)

# Create parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [5, 10, 15]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(als.maxIter, [5, 10, 15]) \
    .build()

total_comb = len(param_grid)
print(f"\nTotal parameter combinations: {total_comb}")
print(f"Grid: 3 ranks × 3 regParams × 3 maxIters = {total_comb} combinations")

# 3. Cross-Validation

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

cv = CrossValidator(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,  # 3-fold cross-validation
    seed=42,
    parallelism=2  # Parallel execution
)

print(f"Cross-validation folds: 3")
print(f"Evaluation metric: RMSE")
print(f"Parallelism: 2")

print(f"\nTraining {total_comb} models with 3-fold CV...")
print("Progress: ", end="", flush=True)

start_time = time.time()
cv_model = cv.fit(train_df)
end_time = time.time()

total_time = end_time - start_time
print(f"\n\nCross-validation complete!")
print(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
print(f"Average time per model: {total_time / total_comb:.2f} seconds")

# 4.  Extract Best Model and Parameters

best_model = cv_model.bestModel
best_rank = best_model.rank
best_regParam = best_model._java_obj.parent().getRegParam()
best_maxIter = best_model._java_obj.parent().getMaxIter()

print(f"\nBest hyperparameters:")
print(f"rank:      {best_rank}")
print(f"regParam:  {best_regParam}")
print(f"maxIter:   {best_maxIter}")

# Evaluate best model on test set
predictions = best_model.transform(test_df)
best_rmse = evaluator.evaluate(predictions)

print(f"\nBest model performance:")
print(f"Test RMSE: {best_rmse:.4f}")

# 5. Analyze All Results

# Get all average metrics from cross-validation
avg_metrics = cv_model.avgMetrics

# Create results list with parameters
results = []
for idx, (params, metric) in enumerate(zip(param_grid, avg_metrics)):
    rank = params[als.rank]
    reg_param = params[als.regParam]
    max_iter = params[als.maxIter]

    results.append({
        'rank': rank,
        'regParam': reg_param,
        'maxIter': max_iter,
        'cv_rmse': metric,
        'is_best': (metric == min(avg_metrics))
    })

# Sort by RMSE
results_sorted = sorted(results, key=lambda x: x['cv_rmse'])

print("\nTop 5 parameter combinations:")
print(f"{'Rank':<6} {'RegParam':<10} {'MaxIter':<9} {'CV RMSE':<10}")
print("-" * 40)
for r in results_sorted[:5]:
    marker = " ←Best" if r['is_best'] else ""
    print(f"{r['rank']:<6} {r['regParam']:<10.2f} {r['maxIter']:<9} {r['cv_rmse']:<10.4f}{marker}")

print("\nWorst 5 parameter combinations:")
print(f"{'Rank':<6} {'RegParam':<10} {'MaxIter':<9} {'CV RMSE':<10}")
print("-" * 40)
for r in results_sorted[-5:]:
    print(f"{r['rank']:<6} {r['regParam']:<10.2f} {r['maxIter']:<9} {r['cv_rmse']:<10.4f}")

# 6. Parameter Impact Analysis

print("\nAnalyze impact of each parameter:")
def avg_by_param(param):
    vals = sorted(set([r[param] for r in results]))
    for v in vals:
        subset = [r["cv_rmse"] for r in results if r[param] == v]
        print(f"  {param}={v:<5}: avg RMSE = {sum(subset)/len(subset):.4f}")


print("\nRank impact:")
avg_by_param("rank")

print("\nRegParam impact:")
avg_by_param("regParam")

print("\nMaxIter impact:")
avg_by_param("maxIter")


# Visualization
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

try:
    #heatmap
    ranks = sorted(set(r['rank'] for r in results))
    regs = sorted(set(r['regParam'] for r in results))
    heat = np.zeros((len(regs), len(ranks)))

    for r in results:
        i = regs.index(r['regParam'])
        j = ranks.index(r['rank'])
        heat[i, j] += r['cv_rmse']


    heat /= len(set(r['maxIter'] for r in results))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heat, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(ranks)))
    ax.set_yticks(range(len(regs)))
    ax.set_xticklabels(ranks)
    ax.set_yticklabels([f"{r:.2f}" for r in regs])
    ax.set_xlabel("Rank (Latent Factors)", fontweight="bold")
    ax.set_ylabel("RegParam (Regularization)", fontweight="bold")
    ax.set_title("RMSE Heatmap: Rank vs RegParam (Avg over MaxIter)", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE", fontweight="bold")

    for i in range(len(regs)):
        for j in range(len(ranks)):
            ax.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("hyperparam_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: hyperparam_heatmap.png")

    # Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    param_colors = ["steelblue", "coral", "green"]
    params = ["rank", "regParam", "maxIter"]

    for idx, param in enumerate(params):
        vals = sorted(set(r[param] for r in results))
        avg_rmse = [np.mean([x["cv_rmse"] for x in results if x[param] == v]) for v in vals]

        axes[idx].bar(vals, avg_rmse, color=param_colors[idx], edgecolor="black", alpha=0.8)
        axes[idx].set_title(f"Impact of {param}", fontweight="bold")
        axes[idx].set_xlabel(param, fontweight="bold")
        axes[idx].set_ylabel("Avg RMSE", fontweight="bold")
        axes[idx].grid(axis="y", alpha=0.3)
        for x, y in zip(vals, avg_rmse):
            axes[idx].text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("parameter_impacts.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: parameter_impacts.png")

except Exception as e:
    print(f"Visualization error: {e}")


spark.stop()
print("\n----- Complete -----\n")
