"""
MIE1628 Assignment 2 - Part B-1
Data Description and Insights Analysis
"""

import os
import sys

# Configure Spark Python environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession


# Initialize Spark Session
spark = SparkSession.builder.appName("MovieRecommender-PartB1").getOrCreate()

# 1. Load Data
df = spark.read.csv("movies.csv",
                    header=True, inferSchema=True)

df.printSchema()
df.show(5)

# 2. Dataset Overview
# check the number of unique users、movies
from pyspark.sql.functions import countDistinct

# Dataset overview
unique_users = df.select(countDistinct("userId")).collect()[0][0]
unique_movies = df.select(countDistinct("movieId")).collect()[0][0]
total_records = df.count()
sparsity = 1 - total_records / (unique_users * unique_movies)
print(f"\nUnique users: {unique_users}")
print(f"Unique movies: {unique_movies}")
print(f"Sparsity: {sparsity*100:.2f}%")


# 3. Rating Distribution Analysis
from pyspark.sql.functions import count, countDistinct, mean, stddev, col, percentile_approx

print("\n","Rating Distribution:")
rating_summary = df.groupBy("rating").agg(count("*").alias("count")).orderBy("rating")
rating_list = rating_summary.collect()
total_count = sum(row["count"] for row in rating_list)

print("\nRating Breakdown with Percentage:")
for row in rating_list:
    rating = row["rating"]
    count_val = row["count"]
    percentage = (count_val / total_count) * 100
    print(f"  Rating {rating:.1f}: {count_val:>5,} ({percentage:>6.2f}%)")

print("\n")
df.describe("rating").show()

stats = df.select(
    mean("rating").alias("mean"),
    stddev("rating").alias("std"),
    percentile_approx("rating", 0.5).alias("median")
)
print("\nRating Summary Statistics:")
stats.show()


# 4. Find Top 10 movies & Top 10 users
print("\n","Find Top 10 movies:")

movie_stats = df.groupBy("movieId") \
    .agg(mean("rating").alias("avg_rating"),
         count("rating").alias("num_ratings")) \
    .filter(col("num_ratings") >= 10) \
    .orderBy(col("avg_rating").desc())

movie_stats.show(10)


print("\nTop 10 Active Users with Their Average Ratings:")

user_activity = df.groupBy("userId") \
    .agg(count("rating").alias("num_ratings"))

user_avg = df.groupBy("userId") \
    .agg(mean("rating").alias("avg_rating"))

user_stats = user_activity.join(user_avg, on="userId", how="inner")

top_users = user_stats.orderBy(col("num_ratings").desc())
top_users.show(10)


print("\nAverage rating among Top 10 Active Users:")

top10_avg_rating = top_users.agg(mean("avg_rating").alias("top10_mean_rating")).collect()[0]["top10_mean_rating"]
print(f"The mean of average ratings for the top 10 users is: {top10_avg_rating:.4f}")


# 5. User Behavior Analysis
print("\nUser Behavior Analysis")


user_counts = df.groupBy("userId").agg(count("rating").alias("num_ratings"))


user_counts_list = [row["num_ratings"] for row in user_counts.collect()]
total_users = len(user_counts_list)

activity_ranges = [
    (1, 20, "Light users (1–20 ratings)"),
    (21, 50, "Moderate users (21–50 ratings)"),
    (51, 100, "Active users (51–100 ratings)"),
    (101, 10000, "Super active users (100+ ratings)")
]

print("\nUser Activity Distribution:")
for min_r, max_r, label in activity_ranges:
    count_users = len([n for n in user_counts_list if min_r <= n <= max_r])
    percentage = (count_users / total_users) * 100
    print(f"  {label:<35} {count_users:>6,} users ({percentage:>5.2f}%)")


avg_ratings_per_user = sum(user_counts_list) / total_users
print(f"\nAverage number of ratings per user: {avg_ratings_per_user:.2f}")

# 6.  Visualization
import matplotlib.pyplot as plt
import pandas as pd


# Convert Spark DataFrames to Pandas for visualization
rating_pd = rating_summary.toPandas()
top_movies_pd = movie_stats.limit(10).toPandas()
top_users_pd = user_activity.limit(10).toPandas()

# Rating Distribution
plt.figure(figsize=(8, 5))
plt.bar(rating_pd["rating"], rating_pd["count"], color="steelblue", edgecolor="black", alpha=0.8)
plt.xlabel("Rating", fontsize=12, fontweight="bold")
plt.ylabel("Count", fontsize=12, fontweight="bold")
plt.title("Distribution of Ratings", fontsize=14, fontweight="bold")
for i, v in enumerate(rating_pd["count"]):
    plt.text(rating_pd["rating"][i], v + 5, str(v), ha="center", fontweight="bold", fontsize=8)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("rating_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: rating_distribution.png")

#Top 10 Movies
plt.figure(figsize=(10, 6))
plt.barh(top_movies_pd["movieId"].astype(str), top_movies_pd["avg_rating"],
         color="coral", edgecolor="black", alpha=0.8)
plt.xlabel("Average Rating", fontsize=12, fontweight="bold")
plt.ylabel("Movie ID", fontsize=12, fontweight="bold")
plt.title("Top 10 Movies by Average Rating", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()  # highest at top
for i, (avg, num) in enumerate(zip(top_movies_pd["avg_rating"], top_movies_pd["num_ratings"])):
    plt.text(avg, i, f" {avg:.2f} ({num} ratings)", va="center", fontweight="bold")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("top_movies_by_rating.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: top_movies_by_rating.png")

#Top 10 Users
plt.figure(figsize=(10, 6))
plt.barh(top_users_pd["userId"].astype(str), top_users_pd["num_ratings"],
         color="seagreen", edgecolor="black", alpha=0.8)
plt.xlabel("Number of Ratings", fontsize=12, fontweight="bold")
plt.ylabel("User ID", fontsize=12, fontweight="bold")
plt.title("Top 10 Most Active Users", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
for i, num in enumerate(top_users_pd["num_ratings"]):
    plt.text(num, i, f" {num}", va="center", fontweight="bold")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("top_active_users.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: top_active_users.png")

# User Activity Distribution
user_counts = [r["num_ratings"] for r in df.groupBy("userId").agg(count("rating").alias("num_ratings")).collect()]
plt.figure(figsize=(8, 5))
plt.hist(user_counts, bins=20, color="purple", alpha=0.7, edgecolor="black")
plt.xlabel("Number of Ratings per User", fontsize=12, fontweight="bold")
plt.ylabel("Number of Users", fontsize=12, fontweight="bold")
plt.title("User Activity Distribution", fontsize=14, fontweight="bold")
plt.axvline(avg_ratings_per_user, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {avg_ratings_per_user:.1f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("user_activity_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: user_activity_distribution.png")
spark.stop()
print("\n----- Complete -----")
