"""
MIE1628 Assignment 2 - Part B-3
In-Depth Evaluation of Error Metrics

"""

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, avg, count, when

# Initialize Spark
spark = SparkSession.builder \
    .appName("MovieRecommender-PartB3") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 1. Load and Prepare Data -----
file_path = "movies.csv"

df = spark.read \
    .format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .load(file_path)

df = df.selectExpr("userId as userId", "movieId as movieId", "rating as rating")
print(f"Total records: {df.count():,}")

# 2. Split Data

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

train_count = train_df.count()
test_count = test_df.count()

print(f"Training set: {train_count:,} records")
print(f"Test set:     {test_count:,} records")

# 3. Train ALS Model
als = ALS(
    rank=10,
    maxIter=10,
    regParam=0.1,
    userCol='userId',
    itemCol='movieId',
    ratingCol='rating',
    coldStartStrategy='drop'
)

model = als.fit(train_df)
model = als.fit(train_df)
predictions = model.transform(test_df).cache()
print(f"Predictions: {predictions.count():,}")

# 4. Regression Metrics

# MSE
mse_evaluator = RegressionEvaluator(
    metricName="mse",
    labelCol="rating",
    predictionCol="prediction"
)
mse = mse_evaluator.evaluate(predictions)

# RMSE
rmse_evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = rmse_evaluator.evaluate(predictions)

# MAE
mae_evaluator = RegressionEvaluator(
    metricName="mae",
    labelCol="rating",
    predictionCol="prediction"
)
mae = mae_evaluator.evaluate(predictions)

print("\nRegression Metrics Results:")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# 5.  Classification Metrics

th = 3.0
print(f"\nUsing threshold: {th} (ratings >= {th} are relevant)")

# Create binary labels: 1 if relevant, 0 if not
pred_bin = (
    predictions.withColumn("actual", when(col("rating") >= th, 1).otherwise(0))
    .withColumn("predicted", when(col("prediction") >= th, 1).otherwise(0))
)

# Calculate confusion matrix components
tp = pred_bin.filter(
    (col("actual") == 1) & (col("predicted") == 1)
).count()

fp = pred_bin.filter(
    (col("actual") == 0) & (col("predicted") == 1)
).count()

tn = pred_bin.filter(
    (col("actual") == 0) & (col("predicted") == 0)
).count()

fn = pred_bin.filter(
    (col("actual") == 1) & (col("predicted") == 0)
).count()

print("\nConfusion Matrix:")
print(f"  True Positives (TP):  {tp}")
print(f"  False Positives (FP): {fp}")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Negatives (FN): {fn}")

# Calculate Precision, Recall, F1
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\nClassification Metrics Results:")
print(f"  • Precision: {precision:.4f}")
print(f"  • Recall:    {recall:.4f}")
print(f"  • F1 Score:  {f1_score:.4f}")

# 6. Summary Table
metrics_summary = {
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score
}

print(f"\n{'Metric':<15} {'Value':<10} {'Type':<20}")
print("-" * 45)
print(f"{'MSE':<15} {mse:<10.4f} {'Regression':<20}")
print(f"{'RMSE':<15} {rmse:<10.4f} {'Regression':<20}")
print(f"{'MAE':<15} {mae:<10.4f} {'Regression':<20}")
print(f"{'Precision':<15} {precision:<10.4f} {'Classification':<20}")
print(f"{'Recall':<15} {recall:<10.4f} {'Classification':<20}")
print(f"{'F1 Score':<15} {f1_score:<10.4f} {'Classification':<20}")

# 7.  Visualization


try:
    import matplotlib.pyplot as plt
    import numpy as np

    # Visualization 1: Regression Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    regression_metrics = ['MSE', 'RMSE', 'MAE']
    regression_values = [mse, rmse, mae]
    colors = ['steelblue', 'coral', 'green']

    bars = ax.bar(regression_metrics, regression_values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Regression Metrics Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars, regression_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('regression_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: regression_metrics.png")

    # Visualization 2: Classification Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    classification_metrics = ['Precision', 'Recall', 'F1 Score']
    classification_values = [precision, recall, f1_score]
    colors = ['purple', 'orange', 'teal']

    bars = ax.bar(classification_metrics, classification_values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('Classification Metrics Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars, classification_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('classification_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: classification_metrics.png")

    # Visualization 3: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    confusion_matrix = np.array([[tp, fp], [fn, tn]])
    im = ax.imshow(confusion_matrix, cmap='Blues', alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'], fontsize=11)
    ax.set_yticklabels(['Actual Positive', 'Actual Negative'], fontsize=11)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="black",
                           fontsize=16, fontweight='bold')

    # Labels
    labels = [['TP', 'FP'], ['FN', 'TN']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.3, labels[i][j],
                    ha="center", va="center", color="red",
                    fontsize=12, fontweight='bold')

    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: confusion_matrix.png")

except ImportError:
    print("Matplotlib not available")
except Exception as e:
    print(f"Visualization error: {e}")



predictions.unpersist()
spark.stop()


print("\n----- Complete -----\n")
