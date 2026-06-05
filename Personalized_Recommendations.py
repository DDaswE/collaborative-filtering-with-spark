"""
MIE1628 Assignment 2 - Part B-5
Personalized Recommendations and Analysis for Selected Users
Generate personalized movie recommendations for users 11 and 21
"""

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, avg, lit

# Initialize Spark
spark = SparkSession.builder.appName("MovieRecommender-PartB5").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 1. Load and Prepare Data
file_path = "movies.csv"
df = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load("movies.csv")
    .select("userId", "movieId", "rating")
)
print(f"Total records: {df.count():,}")

# 2. Split Data and Train Model
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Training set: {train_df.count():,} records")
print(f"Test set:     {test_df.count():,} records")

# Use the best parameters from Part B-4 (or default good parameters)
als = ALS(
    rank=15,
    maxIter=15,
    regParam=0.1,
    userCol='userId',
    itemCol='movieId',
    ratingCol='rating',
    coldStartStrategy='drop',
    seed=42
)
model = als.fit(train_df)
print("Model training complete")

#3. Analyze Target Users

target_users = [11, 21]
user_profiles = []

for user_id in target_users:
    print(f"\n--- User {user_id} Profile ---")

    # Get user's rating history
    user_ratings = df.filter(col("userId") == user_id).orderBy(col("rating").desc())
    num_ratings = user_ratings.count()

    if num_ratings == 0:
        print(f"User {user_id} has no rating history")
        continue

    # Calculate statistics
    user_stats = user_ratings.select("rating").describe().collect()
    avg_rating = user_ratings.select(avg("rating")).collect()[0][0]
    top_movies = user_ratings.orderBy(col("rating").desc()).limit(3).collect()
    print(f"\nUser {user_id} — Ratings: {num_ratings}, Avg Rating: {avg_rating:.2f}")
    print("Top Rated Movies:", ", ".join([f"{r.movieId}({r.rating:.1f})" for r in top_movies]))

    user_profiles.append({"userId": user_id, "num_ratings": num_ratings, "avg_rating": avg_rating})


#4. Generate Personalized Recommendations
num_recs = 10

for user_id in target_users:
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recs = model.recommendForUserSubset(user_df, num_recs).collect()
    if not recs:
        print(f"User {user_id}: No recommendations (cold start).")
        continue

    print(f"\nTop {num_recs} Recommendations for User {user_id}:")
    print(f"{'Rank':<5} {'MovieID':<10} {'Predicted Rating':<15}")
    print("-" * 35)
    for i, rec in enumerate(recs[0]["recommendations"], 1):
        print(f"{i:<5} {rec['movieId']:<10} {rec['rating']:<15.4f}")



# 5. Evaluate Recommendation Quality


# Create baseline recommendations (most popular movies)
print("\nBaseline: Most Popular Movies (by rating count)")

from pyspark.sql.functions import count as spark_count, avg as spark_avg

popular_movies = df.groupBy("movieId") \
    .agg(
        spark_count("rating").alias("rating_count"),
        spark_avg("rating").alias("avg_rating")
    ) \
    .orderBy(col("rating_count").desc()) \
    .limit(10)

print(f"\n{'Rank':<6} {'Movie ID':<10} {'Rating Count':<15} {'Avg Rating':<12}")
print("-" * 45)
for rank, row in enumerate(popular_movies.collect(), 1):
    print(f"{rank:<6} {row.movieId:<10} {row.rating_count:<15} {row.avg_rating:<12.2f}")

# 6. Compare Personalized vs Baseline


for user_id in target_users:
    print(f"\n--- User {user_id} Comparison ---")

    # Get user's rating history
    user_history = df.filter(col("userId") == user_id)

    if user_history.count() == 0:
        continue

    # Get personalized recommendations
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    personalized_recs = model.recommendForUserSubset(user_df, num_recs)

    if personalized_recs.count() == 0:
        print("No personalized recommendations available")
        continue

    personalized_movie_ids = [rec['movieId'] for rec in personalized_recs.collect()[0]['recommendations']]

    # Get popular movies
    popular_movie_ids = [row.movieId for row in popular_movies.collect()]

    # Check overlap with user's rated movies
    rated_movie_ids = [row.movieId for row in user_history.select("movieId").collect()]

    personalized_overlap = len(set(personalized_movie_ids) & set(rated_movie_ids))
    popular_overlap = len(set(popular_movie_ids) & set(rated_movie_ids))

    print(f"\nPersonalized recommendations:")
    print(f"Movies already rated: {personalized_overlap}/{num_recs}")
    print(f"New movie suggestions: {num_recs - personalized_overlap}/{num_recs}")

    print(f"\nPopular (baseline) recommendations:")
    print(f"Movies already rated: {popular_overlap}/{num_recs}")
    print(f"New movie suggestions: {num_recs - popular_overlap}/{num_recs}")

    # Calculate average predicted rating
    avg_predicted = sum(
        [rec['rating'] for rec in personalized_recs.collect()[0]['recommendations']]) / num_recs
    user_avg_rating = user_history.select("rating").agg({"rating": "avg"}).collect()[0][0]

    print(f"\nRating alignment:")
    print(f"User's avg rating:        {user_avg_rating:.2f}")
    print(f"Avg predicted rating:     {avg_predicted:.2f}")
    print(f"Difference:               {abs(user_avg_rating - avg_predicted):.2f}")


#  Visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from pyspark.sql.functions import count as spark_count, avg as spark_avg

    user_data = []

    for user_id in target_users:
        user_ratings = df.filter(col("userId") == user_id)
        if user_ratings.count() == 0:
            continue

        num_ratings = user_ratings.count()
        avg_rating = user_ratings.select("rating").agg({"rating": "avg"}).collect()[0][0]

        # Personalized recommendations
        user_df = spark.createDataFrame([(user_id,)], ["userId"])
        recs = model.recommendForUserSubset(user_df, 10)
        if recs.count() > 0:
            rec_list = recs.collect()[0]['recommendations']
            avg_predicted = sum([rec['rating'] for rec in rec_list]) / len(rec_list)
        else:
            avg_predicted = 0

        user_data.append({
            'user_id': user_id,
            'num_ratings': num_ratings,
            'avg_rating': avg_rating,
            'avg_predicted': avg_predicted
        })

    # Plot 1: Number of Ratings per User
    if len(user_data) > 0:
        users = [d['user_id'] for d in user_data]
        num_ratings = [d['num_ratings'] for d in user_data]

        plt.figure(figsize=(7, 5))
        plt.bar(range(len(users)), num_ratings, color=['steelblue', 'coral'],
                alpha=0.8, edgecolor='black')
        plt.xticks(range(len(users)), [f'User {u}' for u in users])
        plt.ylabel('Number of Ratings', fontsize=11, fontweight='bold')
        plt.title('User Activity Level', fontsize=13, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        for i, count in enumerate(num_ratings):
            plt.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig('user_num_ratings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: user_num_ratings.png")

    # Plot 2: Average Rating Comparison
    if len(user_data) > 0:
        x = range(len(users))
        width = 0.35
        avg_ratings = [d['avg_rating'] for d in user_data]
        avg_predicted = [d['avg_predicted'] for d in user_data]

        plt.figure(figsize=(7, 5))
        plt.bar([i - width / 2 for i in x], avg_ratings, width,
                label='Actual Avg Rating', color='green', alpha=0.8, edgecolor='black')
        plt.bar([i + width / 2 for i in x], avg_predicted, width,
                label='Predicted Avg Rating', color='purple', alpha=0.8, edgecolor='black')
        plt.xticks(x, [f'User {u}' for u in users])
        plt.ylabel('Average Rating', fontsize=11, fontweight='bold')
        plt.title('Average Rating Comparison', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        for i, (actual, pred) in enumerate(zip(avg_ratings, avg_predicted)):
            plt.text(i - width / 2, actual, f'{actual:.2f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
            plt.text(i + width / 2, pred, f'{pred:.2f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.tight_layout()
        plt.savefig('user_avg_rating_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: user_avg_rating_comparison.png")

    # Plot 3: Personalized vs Baseline overlap
    personalized_new = []
    popular_new = []
    labels = []

    for user_id in target_users:
        user_history = df.filter(col("userId") == user_id)
        if user_history.count() == 0:
            continue

        user_df = spark.createDataFrame([(user_id,)], ["userId"])
        personalized_recs = model.recommendForUserSubset(user_df, 10)
        if personalized_recs.count() == 0:
            personalized_new.append(0)
            popular_new.append(0)
            labels.append(f'User {user_id}')
            continue

        personalized_movie_ids = [rec['movieId'] for rec in personalized_recs.collect()[0]['recommendations']]
        popular_movie_ids = [row.movieId for row in popular_movies.collect()]
        rated_movie_ids = [row.movieId for row in user_history.select("movieId").collect()]

        personalized_overlap = len(set(personalized_movie_ids) & set(rated_movie_ids))
        popular_overlap = len(set(popular_movie_ids) & set(rated_movie_ids))

        personalized_new.append(10 - personalized_overlap)
        popular_new.append(10 - popular_overlap)
        labels.append(f'User {user_id}')

    if len(labels) > 0:
        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(7, 5))
        plt.bar(x - width / 2, personalized_new, width, label='Personalized New', color='teal', alpha=0.8)
        plt.bar(x + width / 2, popular_new, width, label='Baseline New', color='grey', alpha=0.6)
        plt.xticks(x, labels)
        plt.ylabel('New (Unseen) Recommendations', fontsize=11, fontweight='bold')
        plt.title('Personalized vs Baseline Recommendation Novelty', fontsize=13, fontweight='bold')
        plt.legend()
        plt.ylim(0, max(max(personalized_new), max(popular_new)) + 1)
        plt.grid(axis='y', alpha=0.3)

        # 数值标签（包括 0）
        for i, val in enumerate(personalized_new):
            plt.text(i - width / 2, val + 0.1, f'{val}', ha='center', color='teal', fontweight='bold')
        for i, val in enumerate(popular_new):
            plt.text(i + width / 2, val + 0.1, f'{val}', ha='center', color='grey', fontweight='bold')

        plt.tight_layout()
        plt.savefig('recommendation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: recommendation_comparison.png")


except ImportError:
    print("Matplotlib not available")
except Exception as e:
    print(f"Visualization error: {e}")




spark.stop()
print("\n----- Complete -----\n")
