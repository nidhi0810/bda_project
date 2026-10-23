"""Schema-aware PySpark analysis for the social media mini project."""
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
import re

ALIASES = {
    "platform": ["platform", "social_platform", "source", "network"],
    "hashtag": ["hashtag", "hashtags", "hash_tag", "tag", "topic", "topic_name"],
    "topic": ["topic", "topic_category", "category", "subject"],
    "sentiment": ["sentiment", "sentiment_label", "sentiment_category", "polarity", "emotion"],
    "timestamp": ["timestamp", "datetime", "date_time", "created_at", "posted_at", "post_time", "date", "time"],
    "content_type": ["content_type", "media_type", "post_type", "format"],
    "region": ["region", "country", "location", "geography", "state"],
    "views": ["views", "view_count", "impressions", "reach"],
    "likes": ["likes", "like_count", "favorite_count", "favourites"],
    "shares": ["shares", "share_count", "retweets", "reposts"],
    "comments": ["comments", "comment_count", "replies"],
    "text": ["text", "post_text", "content", "caption", "message", "tweet"],
    "post_id": ["post_id", "id", "postid", "tweet_id", "content_id"],
}

def create_spark():
    return (SparkSession.builder
            .appName("SocialMediaBigDataMiniProject")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())

def normalized_name(s):
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def detect_columns(columns):
    by_norm = {normalized_name(c): c for c in columns}
    found = {}
    for logical, candidates in ALIASES.items():
        for candidate in candidates:
            if candidate in by_norm:
                found[logical] = by_norm[candidate]
                break
    # A dataset may call the same idea "platform name" or "post timestamp".
    if "platform" not in found:
        for c in columns:
            n = normalized_name(c)
            if "platform" in n or n.endswith("_network"):
                found["platform"] = c
                break
    if "timestamp" not in found:
        for c in columns:
            n = normalized_name(c)
            if "date" in n or "time" in n:
                found["timestamp"] = c
                break
    return found

def load_dataset(spark, path):
    raw = (spark.read.option("header", True)
           .option("inferSchema", True)
           .option("multiLine", True)
           .option("escape", '"')
           .csv(path))
    if len(raw.columns) == 0:
        raise ValueError("The CSV appears to have no columns. Check that the correct CSV was selected.")
    mapping = detect_columns(raw.columns)
    df = raw
    # Standardized helper columns make the rest of the dashboard consistent.
    for logical, source in mapping.items():
        if source != logical:
            df = df.withColumnRenamed(source, logical)
    mapping = detect_columns(df.columns)

    # Convert known engagement metrics to numeric where available.
    for c in ["views", "likes", "shares", "comments"]:
        if c in df.columns:
            df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), ",", "").cast("double"))
    if "timestamp" in df.columns:
        df = df.withColumn("timestamp", F.coalesce(
            F.to_timestamp(F.col("timestamp")),
            F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd"),
            F.to_timestamp(F.col("timestamp"), "MM/dd/yyyy HH:mm:ss"),
            F.to_timestamp(F.col("timestamp"), "MM/dd/yyyy")
        ))
        df = df.withColumn("posting_hour", F.hour("timestamp"))
        df = df.withColumn("posting_date", F.to_date("timestamp"))

    # If hashtags/topics aren't provided, leave a clear fallback instead of fabricating labels.
    if "hashtag" not in df.columns and "topic" in df.columns:
        df = df.withColumn("hashtag", F.col("topic").cast("string"))
    if "topic" not in df.columns and "hashtag" in df.columns:
        df = df.withColumn("topic", F.col("hashtag").cast("string"))

    before = df.count()
    if "post_id" in df.columns:
        df = df.dropDuplicates(["post_id"])
    else:
        df = df.dropDuplicates()
    after_dedup = df.count()

    # Keep missing categorical values but label them explicitly for charting.
    for c in ["platform", "hashtag", "topic", "sentiment", "content_type", "region"]:
        if c in df.columns:
            df = df.withColumn(c, F.coalesce(F.col(c).cast("string"), F.lit("Unknown")))
    return df, mapping, {"rows_before_dedup": before, "rows_after_dedup": after_dedup,
                         "duplicates_removed": before - after_dedup, "columns": raw.columns}

def dataframe_to_pandas(df, limit=5000):
    return df.limit(limit).toPandas()

def get_summary(df):
    numeric = [c for c in ["views", "likes", "shares", "comments"] if c in df.columns]
    expressions = [F.count("*").alias("row_count")]
    for c in numeric:
        expressions.append(F.sum(F.col(c)).alias("total_" + c))
        expressions.append(F.avg(F.col(c)).alias("avg_" + c))
    return df.agg(*expressions).toPandas()

def grouped_counts(df, column, count_name="posts"):
    return (df.groupBy(column).agg(F.count("*").alias(count_name))
            .orderBy(F.desc(count_name)))

def engagement_by(df, column):
    metrics = [F.count("*").alias("posts")]
    for c in ["views", "likes", "shares", "comments"]:
        if c in df.columns:
            metrics.append(F.avg(F.col(c)).alias("avg_" + c))
            metrics.append(F.sum(F.col(c)).alias("total_" + c))
    if all(c in df.columns for c in ["likes", "shares", "comments"]):
        df = df.withColumn("engagements", F.col("likes") + F.col("shares") + F.col("comments"))
        metrics.append(F.avg("engagements").alias("avg_engagements"))
        metrics.append(F.sum("engagements").alias("total_engagements"))
    return df.groupBy(column).agg(*metrics).orderBy(F.desc("posts"))

def get_time_trend(df):
    if "posting_date" not in df.columns:
        return None
    return df.groupBy("posting_date").agg(F.count("*").alias("posts")).orderBy("posting_date")

def model_classification(df):
    """Optional ML. Uses sentiment labels when available; otherwise derives engagement bands.
    Predictors exclude raw engagement metrics to avoid leakage when engagement is the target.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, RegexTokenizer, HashingTF, IDF
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    work = df
    label_col = None
    task = None
    if "sentiment" in work.columns and work.select("sentiment").distinct().count() >= 2:
        label_col = "sentiment"
        task = "Sentiment classification"
    elif all(c in work.columns for c in ["likes", "shares", "comments"]):
        work = work.withColumn("engagement_total", F.col("likes") + F.col("shares") + F.col("comments"))
        quantiles = work.approxQuantile("engagement_total", [0.33, 0.66], 0.01)
        if len(quantiles) < 2 or quantiles[0] == quantiles[1]:
            return {"available": False, "reason": "Engagement values do not provide at least three usable bands."}
        work = work.withColumn("label", F.when(F.col("engagement_total") <= quantiles[0], "Low")
                               .when(F.col("engagement_total") <= quantiles[1], "Medium").otherwise("High"))
        label_col = "label"
        task = "Engagement category classification (Low/Medium/High)"
    else:
        return {"available": False, "reason": "No usable sentiment label or complete likes/shares/comments metrics were detected."}

    # Use available metadata fields; never use likes/views/shares/comments as predictors.
    categorical = [c for c in ["platform", "content_type", "topic", "region", "hashtag"] if c in work.columns]
    if "posting_hour" in work.columns:
        categorical.append("posting_hour")
    if not categorical:
        return {"available": False, "reason": "No suitable metadata predictors were detected."}

    # Keep only rows with a target; fill missing categories consistently.
    work = work.filter(F.col(label_col).isNotNull())
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec", handleInvalid="keep") for c in categorical]
    assembler = VectorAssembler(inputCols=[f"{c}_vec" for c in categorical], outputCol="features", handleInvalid="keep")
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label_idx", handleInvalid="keep")
    rf = RandomForestClassifier(labelCol="label_idx", featuresCol="features", numTrees=40, seed=42)
    pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer, rf])
    train, test = work.randomSplit([0.8, 0.2], seed=42)
    if train.count() < 20 or test.count() < 5:
        return {"available": False, "reason": "Too few rows for a meaningful train/test split (need at least 25 rows)."}
    model = pipeline.fit(train)
    pred = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol="label_idx", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(pred)
    f1eval = MulticlassClassificationEvaluator(labelCol="label_idx", predictionCol="prediction", metricName="f1")
    f1 = f1eval.evaluate(pred)
    return {
        "available": True, "task": task, "label": label_col, "predictors": categorical,
        "test_rows": test.count(), "accuracy": accuracy, "f1": f1,
        "predictions": pred.select(label_col, "prediction", "probability").limit(100).toPandas(),
        "reason": None
    }
