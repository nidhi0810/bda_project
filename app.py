from pathlib import Path
import tempfile

import streamlit as st
import pandas as pd

from analytics import (
    create_spark,
    load_dataset,
    grouped_counts,
    engagement_by,
    get_time_trend,
)

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer,
    IndexToString,
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Social Media Big Data Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Social Media Trend Detection & Sentiment Analytics")
st.caption("Big Data Mini Project | PySpark + Streamlit + Random Forest")

st.markdown("""
Explore social media trends across platforms, topics, sentiment, engagement,
and time. Train a Random Forest model that predicts a post's relative
virality category (Low / Medium / High).
""")

DEFAULT = Path("data/multi_platform_social_sentiment_evolution.csv")


# =========================================================
# SPARK SESSION
# =========================================================

@st.cache_resource
def spark_session():
    return create_spark()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_column(columns, candidates):
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def safe_sum(spark_df, column):
    result = spark_df.agg(
        F.sum(F.col(column).cast("double")).alias("total")
    ).first()["total"]
    return result if result is not None else 0


def spark_preview(spark_df, n=10):
    return spark_df.limit(n).toPandas()


def calculate_class_weights(train_df, target_column):
    """Give less frequent classes more influence during training."""
    rows = train_df.groupBy(target_column).count().collect()
    counts = {str(r[target_column]): int(r["count"]) for r in rows}
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {
        label: total / (n_classes * count) for label, count in counts.items()
    }
    weight_expr = F.lit(1.0)
    for label, weight in weights.items():
        weight_expr = F.when(
            F.col(target_column) == F.lit(label),
            F.lit(float(weight)),
        ).otherwise(weight_expr)
    return train_df.withColumn("classWeight", weight_expr), counts, weights


def get_numeric_default(spark_df, column):
    values = spark_df.approxQuantile(column, [0.5], 0.01)
    if values and values[0] is not None:
        return float(values[0])
    return 0.0


# =========================================================
# FEATURE DEFINITIONS (used for training AND prediction)
# =========================================================

# Known when (or before) a post is published.
PRE_NUMERIC = [
    "hour_of_day", "day_of_week", "is_weekend", "followers",
    "account_age_days", "verified", "content_length", "num_hashtags",
    "toxicity_score", "sentiment_positive", "sentiment_negative",
    "sentiment_neutral",
]

# Observed AFTER publishing (early engagement signals).
EARLY_NUMERIC = ["hours_since_post", "likes", "comments", "shares"]

CATEGORICAL = [
    "platform", "topic", "language", "media_type",
    "sentiment_category", "location",
]

# name -> (required raw columns, expression builder)
DERIVED = {
    "log_followers": (
        ["followers"],
        lambda: F.log1p(F.col("followers")),
    ),
    "followers_per_day": (
        ["followers", "account_age_days"],
        lambda: F.col("followers") / (F.col("account_age_days") + 1),
    ),
    "hashtag_density": (
        ["num_hashtags", "content_length"],
        lambda: F.col("num_hashtags") / (F.col("content_length") + 1),
    ),
    "log_likes": (["likes"], lambda: F.log1p(F.col("likes"))),
    "log_comments": (["comments"], lambda: F.log1p(F.col("comments"))),
    "log_shares": (["shares"], lambda: F.log1p(F.col("shares"))),
    "shares_per_follower": (
        ["shares", "followers"],
        lambda: F.col("shares") / (F.col("followers") + 1),
    ),
    "eng_per_follower": (
        ["likes", "comments", "shares", "followers"],
        lambda: (F.col("likes") + F.col("comments") + F.col("shares"))
        / (F.col("followers") + 1),
    ),
    "eng_per_hour": (
        ["likes", "comments", "shares", "hours_since_post"],
        lambda: (F.col("likes") + F.col("comments") + F.col("shares"))
        / F.greatest(F.col("hours_since_post"), F.lit(0.1)),
    ),
}


def add_derived_features(spark_df, raw_numeric):
    """Add engineered features whose raw inputs are all available.
    Returns (dataframe, list_of_derived_column_names)."""
    created = []
    for name, (needs, builder) in DERIVED.items():
        if all(n in raw_numeric for n in needs):
            spark_df = spark_df.withColumn(name, builder())
            created.append(name)
    return spark_df, created


# =========================================================
# SIDEBAR AND DATASET
# =========================================================

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload the Kaggle CSV", type=["csv"])
    st.caption("Expected: multi_platform_social_sentiment_evolution.csv")
    st.divider()
    st.write("Processing: PySpark local mode")
    st.write("Classifier: Random Forest")


if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getvalue())
        dataset_path = tmp.name
else:
    dataset_path = str(DEFAULT)


if not Path(dataset_path).exists():
    st.warning(
        "Dataset not found. Upload your CSV or put it inside the project's "
        "data folder."
    )
    st.code("data/multi_platform_social_sentiment_evolution.csv")
    st.stop()


try:
    spark = spark_session()
    with st.spinner("Loading dataset with PySpark..."):
        df, mapping, quality = load_dataset(spark, dataset_path)
        row_count = df.count()
except Exception as exc:
    st.error(f"Could not read/process this file: {exc}")
    st.info(
        "Check the CSV path and your Java/PySpark installation. "
        "If a Pandas conversion fails, check your Python dependencies."
    )
    st.stop()


# =========================================================
# 1. DATA QUALITY
# =========================================================

st.subheader("1. Data Quality and Detected Schema")

c1, c2, c3 = st.columns(3)
c1.metric("Rows after deduplication", f"{quality['rows_after_dedup']:,}")
c2.metric("Duplicates removed", f"{quality['duplicates_removed']:,}")
c3.metric("Columns detected", len(quality["columns"]))

with st.expander("Show source columns, field mapping and sample records"):
    st.write("Source columns:", quality["columns"])
    st.write("Detected fields:", mapping)
    st.write("Standardized columns:", df.columns)
    st.dataframe(spark_preview(df, 10), use_container_width=True)


# =========================================================
# FILTERS
# =========================================================

filtered = df
filter_columns = st.columns(3)

filter_candidates = [
    ("platform", ["platform"]),
    ("sentiment", ["sentiment", "sentiment_category"]),
    ("region", ["region", "location"]),
]

for i, (filter_label, candidates) in enumerate(filter_candidates):
    actual_column = find_column(df.columns, candidates)
    if actual_column:
        values = [
            row[0]
            for row in (
                df.select(actual_column)
                .where(F.col(actual_column).isNotNull())
                .distinct()
                .orderBy(actual_column)
                .collect()
            )
        ]
        selected = filter_columns[i].multiselect(
            f"Filter by {filter_label}", values, default=values,
        )
        if selected:
            filtered = filtered.filter(F.col(actual_column).isin(selected))
        else:
            filtered = filtered.filter(F.lit(False))


# =========================================================
# 2. DATASET OVERVIEW
# =========================================================

st.subheader("2. Dataset Overview")

kpis = st.columns(4)
kpis[0].metric("Posts / records", f"{filtered.count():,}")

for idx, (label, candidates) in enumerate(
    [("Views", ["views"]), ("Likes", ["likes"]), ("Shares", ["shares"])],
    start=1,
):
    actual_column = find_column(filtered.columns, candidates)
    if actual_column:
        total = safe_sum(filtered, actual_column)
        kpis[idx].metric(f"Total {label}", f"{total:,.0f}")
    else:
        kpis[idx].metric(label, "Not available")


# =========================================================
# 3. PLATFORM AND TOPIC ANALYSIS
# =========================================================

st.subheader("3. Platform and Topic Analysis")

left, right = st.columns(2)

with left:
    platform_column = find_column(filtered.columns, ["platform"])
    if platform_column:
        platform_pd = grouped_counts(filtered, platform_column).toPandas()
        st.markdown("**Records by platform**")
        if not platform_pd.empty:
            st.bar_chart(platform_pd.set_index(platform_column)["posts"])
            st.download_button(
                "Download platform counts CSV",
                platform_pd.to_csv(index=False),
                "platform_counts.csv",
                "text/csv",
            )
    else:
        st.info("No platform column detected.")

with right:
    topic_column = find_column(
        filtered.columns, ["topic", "hashtag", "content_type"],
    )
    if topic_column:
        topic_pd = grouped_counts(filtered, topic_column).limit(15).toPandas()
        st.markdown(f"**Top 15 {topic_column} categories**")
        if not topic_pd.empty:
            st.bar_chart(topic_pd.set_index(topic_column)["posts"])
            st.download_button(
                "Download topic counts CSV",
                topic_pd.to_csv(index=False),
                "topic_counts.csv",
                "text/csv",
            )
    else:
        st.info("No topic/category column detected.")


# =========================================================
# 4. SENTIMENT ANALYSIS
# =========================================================

st.subheader("4. Sentiment Analysis")

sentiment_column = find_column(
    filtered.columns, ["sentiment", "sentiment_category"],
)

if sentiment_column:
    sentiment_pd = grouped_counts(filtered, sentiment_column).toPandas()
    if not sentiment_pd.empty:
        st.bar_chart(sentiment_pd.set_index(sentiment_column)["posts"])
        st.dataframe(sentiment_pd, use_container_width=True)
        st.download_button(
            "Download sentiment counts CSV",
            sentiment_pd.to_csv(index=False),
            "sentiment_counts.csv",
            "text/csv",
        )
else:
    st.info("No sentiment label detected.")


# =========================================================
# 5. ENGAGEMENT ANALYSIS
# =========================================================

st.subheader("5. Engagement Analysis")

engagement_group = find_column(
    filtered.columns,
    ["platform", "content_type", "topic", "region", "location"],
)
engagement_columns = [
    c for c in ["views", "likes", "shares", "comments"]
    if c in filtered.columns
]

if engagement_group and engagement_columns:
    engagement_pd = engagement_by(filtered, engagement_group).toPandas()
    st.dataframe(engagement_pd, use_container_width=True)
    st.download_button(
        "Download engagement summary CSV",
        engagement_pd.to_csv(index=False),
        "engagement_summary.csv",
        "text/csv",
    )
else:
    st.info("No suitable grouping field or engagement metrics found.")


# =========================================================
# 6. POSTING-TIME TREND
# =========================================================

st.subheader("6. Posting-Time Trend")

try:
    time_pd = get_time_trend(filtered)
    if time_pd is not None:
        time_pd = time_pd.toPandas()
        if not time_pd.empty:
            date_column = find_column(
                time_pd.columns.tolist(),
                ["posting_date", "date", "timestamp"],
            )
            count_column = find_column(
                time_pd.columns.tolist(), ["posts", "count"],
            )
            if date_column and count_column:
                time_pd[date_column] = pd.to_datetime(
                    time_pd[date_column], errors="coerce",
                )
                time_pd = time_pd.dropna(subset=[date_column])
                time_pd = time_pd.sort_values(date_column)
                st.line_chart(time_pd.set_index(date_column)[count_column])
                st.download_button(
                    "Download daily trend CSV",
                    time_pd.to_csv(index=False),
                    "daily_post_trend.csv",
                    "text/csv",
                )
            else:
                st.info("Unexpected columns returned by the time-trend helper.")
        else:
            st.info("No time-trend records are available.")
    else:
        st.info("No recognizable date/timestamp column detected.")
except Exception as exc:
    st.warning(f"Could not generate the time trend: {exc}")


# =========================================================
# 7. MACHINE LEARNING: VIRALITY PREDICTION
# =========================================================

st.subheader("7. Machine Learning: Virality Prediction")

st.markdown("""
The model predicts whether a post has Low, Medium, or High virality. The
categories come from the 33rd and 67th percentiles of `viral_coefficient`,
which itself is never used as an input.
""")

mode = st.radio(
    "Prediction mode",
    ["Early-engagement mode (recommended, ~97% accuracy)",
     "Pre-publication mode (~62-65% accuracy)"],
    help=(
        "Early-engagement mode adds likes, comments, shares and hours since "
        "posting, i.e. what you can observe a few hours after publishing. "
        "Pre-publication mode uses only information known before posting; "
        "virality is largely unpredictable from that alone."
    ),
)
use_early = mode.startswith("Early")

with st.expander("How virality is calculated and why two modes", expanded=False):
    st.write("""
    1. Read the historical viral_coefficient values.
    2. Compute the 33rd and 67th percentile thresholds.
    3. Low: at or below the 33rd percentile. Medium: between the 33rd and
       67th. High: above the 67th.
    4. Train a Random Forest to predict these categories.

    **Why two modes:** in this dataset viral_coefficient is calculated from
    engagement (likes, comments, shares). Before a post is published only
    weak signals exist (followers, platform, timing), so accuracy plateaus
    near 65% with any algorithm. Once early engagement is observed, the
    category becomes very predictable. Use pre-publication mode to estimate
    potential, and early-engagement mode to classify a post that is already
    live.
    """)

if "viral_model" not in st.session_state:
    st.session_state.viral_model = None


if st.button("Train / evaluate virality model", type="primary"):

    viral_column = find_column(filtered.columns, ["viral_coefficient"])

    if viral_column is None:
        st.error(
            "The dataset does not contain viral_coefficient. "
            "Please check the uploaded CSV."
        )
    else:
        wanted_numeric = PRE_NUMERIC + (EARLY_NUMERIC if use_early else [])

        raw_numeric = [
            a for c in wanted_numeric
            if (a := find_column(filtered.columns, [c]))
        ]
        categorical_columns = [
            a for c in CATEGORICAL
            if (a := find_column(filtered.columns, [c]))
        ]

        if not raw_numeric and not categorical_columns:
            st.error("No usable features were found.")
        else:
            try:
                target_column = "virality_category"

                selected_columns = list(dict.fromkeys(
                    raw_numeric + categorical_columns + [viral_column]
                ))

                with st.spinner("Preparing virality categories..."):
                    model_df = filtered.select(*selected_columns)

                    model_df = model_df.withColumn(
                        viral_column, F.col(viral_column).cast("double"),
                    )
                    model_df = model_df.filter(
                        F.col(viral_column).isNotNull()
                        & (~F.isnan(F.col(viral_column)))
                    )

                    for col_name in raw_numeric:
                        model_df = model_df.withColumn(
                            col_name, F.col(col_name).cast("double"),
                        )

                    for col_name in categorical_columns:
                        model_df = model_df.withColumn(
                            col_name,
                            F.coalesce(
                                F.col(col_name).cast("string"),
                                F.lit("Unknown"),
                            ),
                        )

                    usable_rows = model_df.count()
                    if usable_rows < 10:
                        st.error(
                            "There are too few records with a valid "
                            "viral_coefficient to train the model."
                        )
                        st.stop()

                    q33, q67 = model_df.approxQuantile(
                        viral_column, [0.33, 0.67], 0.001,
                    )

                    st.write(f"33rd percentile threshold: {q33:.6f}")
                    st.write(f"67th percentile threshold: {q67:.6f}")

                    if q33 >= q67:
                        st.error(
                            "The virality thresholds are equal or reversed. "
                            "viral_coefficient may contain too many repeated "
                            "values to form three distinct categories."
                        )
                        st.stop()

                    model_df = model_df.withColumn(
                        target_column,
                        F.when(F.col(viral_column) <= F.lit(q33), F.lit("Low"))
                        .when(F.col(viral_column) <= F.lit(q67), F.lit("Medium"))
                        .otherwise(F.lit("High")),
                    )

                    # The raw target must never be a feature.
                    model_df = model_df.drop(viral_column)

                    # Engineered features (log scale, ratios, rates).
                    model_df, derived_columns = add_derived_features(
                        model_df, raw_numeric,
                    )

                    class_counts_pd = (
                        model_df.groupBy(target_column).count()
                        .orderBy(target_column).toPandas()
                    )
                    present = set(class_counts_pd[target_column].astype(str))

                    st.markdown("**Generated virality categories**")
                    st.dataframe(class_counts_pd, use_container_width=True)

                    if present != {"Low", "Medium", "High"}:
                        st.error(
                            "The data did not produce all three virality "
                            "categories. Check the distribution of "
                            "viral_coefficient."
                        )
                        st.stop()

                    train_df, test_df = model_df.randomSplit(
                        [0.8, 0.2], seed=42,
                    )
                    train_df = train_df.cache()
                    test_df = test_df.cache()

                    train_count = train_df.count()
                    test_count = test_df.count()

                    if train_count == 0 or test_count == 0:
                        st.error("The training or test split is empty.")
                        st.stop()

                    training_classes = {
                        row[target_column]
                        for row in train_df.select(target_column)
                        .distinct().collect()
                    }
                    if training_classes != {"Low", "Medium", "High"}:
                        st.error(
                            "All three virality categories must be present "
                            "in the training split. Try another split."
                        )
                        st.stop()

                    train_df, training_class_counts, class_weights = (
                        calculate_class_weights(train_df, target_column)
                    )

                    st.write("Training category counts:", training_class_counts)
                    st.write("Applied class weights:", class_weights)

                    all_numeric = raw_numeric + derived_columns

                    usable_numeric = [
                        c for c in all_numeric
                        if train_df.filter(
                            F.col(c).isNotNull() & (~F.isnan(F.col(c)))
                        ).limit(1).count() > 0
                    ]
                    usable_categorical = [
                        c for c in categorical_columns
                        if train_df.filter(
                            F.col(c).isNotNull()
                        ).limit(1).count() > 0
                    ]

                    if not usable_numeric and not usable_categorical:
                        st.error("All candidate features are unusable.")
                        st.stop()

                # ---------------------------------------------
                # BUILD RANDOM FOREST PIPELINE
                # ---------------------------------------------

                stages = [
                    StringIndexer(
                        inputCol=target_column,
                        outputCol="label",
                        handleInvalid="error",
                    )
                ]

                indexed_columns = []
                for col_name in usable_categorical:
                    index_col = f"{col_name}_index"
                    stages.append(
                        StringIndexer(
                            inputCol=col_name,
                            outputCol=index_col,
                            handleInvalid="keep",
                        )
                    )
                    indexed_columns.append(index_col)

                encoded_columns = []
                if indexed_columns:
                    encoded_columns = [
                        f"{c}_vec" for c in usable_categorical
                    ]
                    stages.append(
                        OneHotEncoder(
                            inputCols=indexed_columns,
                            outputCols=encoded_columns,
                            handleInvalid="keep",
                        )
                    )

                imputed_columns = []
                if usable_numeric:
                    imputed_columns = [f"{c}_imputed" for c in usable_numeric]
                    stages.append(
                        Imputer(
                            inputCols=usable_numeric,
                            outputCols=imputed_columns,
                            strategy="median",
                        )
                    )

                stages.append(
                    VectorAssembler(
                        inputCols=imputed_columns + encoded_columns,
                        outputCol="features",
                        handleInvalid="skip",
                    )
                )

                stages.append(
                    RandomForestClassifier(
                        featuresCol="features",
                        labelCol="label",
                        predictionCol="prediction",
                        probabilityCol="probability",
                        numTrees=150,
                        maxDepth=12,
                        maxBins=64,
                        seed=42,
                        weightCol="classWeight",
                    )
                )

                pipeline = Pipeline(stages=stages)

                with st.spinner("Training the Random Forest model..."):
                    model = pipeline.fit(train_df)

                # ---------------------------------------------
                # EVALUATE THE MODEL
                # ---------------------------------------------

                with st.spinner("Evaluating the held-out test records..."):
                    predictions = model.transform(test_df).cache()
                    evaluated_count = predictions.count()

                    if evaluated_count == 0:
                        st.error("No test records remained after preprocessing.")
                        st.stop()

                    accuracy = MulticlassClassificationEvaluator(
                        labelCol="label",
                        predictionCol="prediction",
                        metricName="accuracy",
                    ).evaluate(predictions)

                    weighted_f1 = MulticlassClassificationEvaluator(
                        labelCol="label",
                        predictionCol="prediction",
                        metricName="f1",
                    ).evaluate(predictions)

                    label_model = model.stages[0]

                    decoder = IndexToString(
                        inputCol="prediction",
                        outputCol="predicted_virality",
                        labels=label_model.labels,
                    )
                    predictions = decoder.transform(predictions)

                # ---------------------------------------------
                # SAVE MODEL AND FORM SETTINGS
                # ---------------------------------------------

                st.session_state.viral_model = model
                st.session_state.viral_target_column = target_column
                # The form only collects RAW inputs; derived ones are
                # recomputed from them at prediction time.
                st.session_state.viral_numeric_columns = raw_numeric
                st.session_state.viral_categorical_columns = usable_categorical
                st.session_state.viral_target_labels = list(label_model.labels)
                st.session_state.viral_target_placeholder = str(
                    class_counts_pd.sort_values("count", ascending=False)
                    .iloc[0][target_column]
                )

                predictor_options = {}
                for col_name in usable_categorical:
                    predictor_options[col_name] = [
                        str(row[0])
                        for row in model_df.select(col_name)
                        .where(F.col(col_name).isNotNull())
                        .distinct().orderBy(col_name).limit(200).collect()
                    ] or ["Unknown"]

                predictor_medians = {
                    c: get_numeric_default(train_df, c) for c in raw_numeric
                }

                st.session_state.viral_options = predictor_options
                st.session_state.viral_medians = predictor_medians
                st.session_state.viral_q33 = float(q33)
                st.session_state.viral_q67 = float(q67)

                st.success("Virality model training and evaluation completed.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Training records", f"{train_count:,}")
                m2.metric("Test records evaluated", f"{evaluated_count:,}")
                m3.metric("Test accuracy", f"{accuracy:.2%}")

                st.metric("Weighted F1-score", f"{weighted_f1:.3f}")

                st.caption(
                    "Features used: "
                    + ", ".join(usable_numeric + usable_categorical)
                )

                st.markdown("**Actual vs predicted virality**")
                result_pd = predictions.select(
                    target_column, "predicted_virality",
                ).toPandas()
                st.dataframe(result_pd.head(100), use_container_width=True)
                st.download_button(
                    "Download virality predictions CSV",
                    result_pd.to_csv(index=False),
                    "virality_predictions.csv",
                    "text/csv",
                )

                st.markdown("**Predicted virality category distribution**")
                distribution_pd = (
                    predictions.groupBy("predicted_virality").count()
                    .orderBy("predicted_virality").toPandas()
                )
                st.bar_chart(
                    distribution_pd.set_index("predicted_virality")["count"]
                )

                st.markdown("**Confusion matrix**")
                confusion_pd = (
                    predictions.groupBy(target_column, "predicted_virality")
                    .count().toPandas()
                )
                confusion_matrix = confusion_pd.pivot(
                    index=target_column,
                    columns="predicted_virality",
                    values="count",
                ).fillna(0).astype(int)
                st.dataframe(confusion_matrix, use_container_width=True)

                st.markdown("**Per-category evaluation**")
                metric_rows = []
                for category in ["Low", "Medium", "High"]:
                    tp = int(
                        confusion_matrix.loc[category, category]
                        if category in confusion_matrix.index
                        and category in confusion_matrix.columns else 0
                    )
                    actual_count = int(
                        confusion_matrix.loc[category].sum()
                        if category in confusion_matrix.index else 0
                    )
                    predicted_count = int(
                        confusion_matrix[category].sum()
                        if category in confusion_matrix.columns else 0
                    )
                    recall = tp / actual_count if actual_count else 0.0
                    precision = tp / predicted_count if predicted_count else 0.0
                    class_f1 = (
                        2 * precision * recall / (precision + recall)
                        if precision + recall else 0.0
                    )
                    metric_rows.append({
                        "Virality category": category,
                        "Precision": round(precision, 4),
                        "Recall": round(recall, 4),
                        "F1-score": round(class_f1, 4),
                        "Actual records": actual_count,
                        "Predicted records": predicted_count,
                    })
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

                st.info(
                    "Categories are relative to this dataset. Results are "
                    "estimates, not guarantees of future popularity."
                )

            except Exception as exc:
                st.error(f"Virality model training failed: {exc}")
                st.exception(exc)


# =========================================================
# 8. INTERACTIVE VIRALITY PREDICTOR
# =========================================================

st.subheader("8. Try the Model: Predict Post Virality")

if st.session_state.viral_model is None:
    st.info(
        "Train the virality model in Section 7 first. "
        "The prediction form will then become available."
    )
else:
    numeric_cols = st.session_state.viral_numeric_columns
    categorical_cols = st.session_state.viral_categorical_columns
    predictor_options = st.session_state.viral_options
    predictor_medians = st.session_state.viral_medians
    target_column = st.session_state.viral_target_column

    label_map = {
        "hour_of_day": "Hour of day (0-23)",
        "day_of_week": "Day of week (0-6)",
        "is_weekend": "Weekend?",
        "followers": "Follower count",
        "account_age_days": "Account age in days",
        "verified": "Verified account?",
        "content_length": "Content length",
        "num_hashtags": "Number of hashtags",
        "toxicity_score": "Toxicity score (0-100)",
        "sentiment_positive": "Positive sentiment (0-1)",
        "sentiment_negative": "Negative sentiment (0-1)",
        "sentiment_neutral": "Neutral sentiment (0-1)",
        "hours_since_post": "Hours since posting",
        "likes": "Likes so far",
        "comments": "Comments so far",
        "shares": "Shares so far",
    }

    with st.form("virality_prediction_form"):
        st.markdown("**Post and account details**")

        form_values = {}
        form_columns = st.columns(2)

        for i, col_name in enumerate(numeric_cols):
            default_value = predictor_medians.get(col_name, 0.0)
            label = label_map.get(col_name, col_name)

            with form_columns[i % 2]:
                if col_name == "hour_of_day":
                    value = st.number_input(
                        label, 0.0, 23.0,
                        float(min(23, max(0, default_value))), 1.0,
                    )
                elif col_name == "day_of_week":
                    value = st.number_input(
                        label, 0.0, 6.0,
                        float(min(6, max(0, default_value))), 1.0,
                    )
                elif col_name in ["is_weekend", "verified"]:
                    value = st.selectbox(
                        label,
                        options=[0.0, 1.0],
                        format_func=lambda x: "Yes" if x == 1.0 else "No",
                    )
                elif col_name.startswith("sentiment_"):
                    value = st.number_input(
                        label, 0.0, 1.0,
                        float(min(1.0, max(0.0, default_value))), 0.01,
                    )
                elif col_name == "toxicity_score":
                    value = st.number_input(
                        label, 0.0, 100.0,
                        float(min(100.0, max(0.0, default_value))), 1.0,
                    )
                elif col_name == "hours_since_post":
                    value = st.number_input(
                        label, 0.1, 48.0,
                        float(min(48.0, max(0.1, default_value))), 0.5,
                    )
                else:
                    value = st.number_input(
                        label, min_value=0.0,
                        value=max(0.0, default_value), step=1.0,
                    )

                form_values[col_name] = float(value)

        st.markdown("**Content and platform details**")
        cat_columns = st.columns(2)

        for i, col_name in enumerate(categorical_cols):
            options = [
                str(x) for x in predictor_options.get(col_name, [])
                if x is not None
            ] or ["Unknown"]
            with cat_columns[i % 2]:
                form_values[col_name] = st.selectbox(
                    col_name.replace("_", " ").title(), options=options,
                )

        submitted = st.form_submit_button("Predict virality", type="primary")

    if submitted:
        try:
            input_columns = list(dict.fromkeys(numeric_cols + categorical_cols))
            prediction_df = df.select(*input_columns).limit(1)

            for col_name in numeric_cols:
                prediction_df = prediction_df.withColumn(
                    col_name,
                    F.lit(float(form_values[col_name])).cast("double"),
                )
            for col_name in categorical_cols:
                prediction_df = prediction_df.withColumn(
                    col_name,
                    F.lit(str(form_values[col_name])).cast("string"),
                )

            # Recompute engineered features exactly as in training.
            prediction_df, _ = add_derived_features(prediction_df, numeric_cols)

            # The fitted pipeline contains a target StringIndexer.
            prediction_df = prediction_df.withColumn(
                target_column,
                F.lit(st.session_state.viral_target_placeholder),
            )

            result = st.session_state.viral_model.transform(prediction_df)
            labels = st.session_state.viral_target_labels

            decoder = IndexToString(
                inputCol="prediction",
                outputCol="predicted_virality",
                labels=labels,
            )
            result = decoder.transform(result)

            result_row = result.select("predicted_virality", "probability").first()
            predicted_category = str(result_row["predicted_virality"])
            probabilities = result_row["probability"].toArray().tolist()
            predicted_probability = float(
                probabilities[labels.index(predicted_category)]
            )

            st.divider()
            st.markdown("### Virality prediction")

            if predicted_category == "High":
                st.success("The model predicts HIGH relative virality.")
            elif predicted_category == "Medium":
                st.info("The model predicts MEDIUM relative virality.")
            else:
                st.warning("The model predicts LOW relative virality.")

            r1, r2 = st.columns(2)
            r1.metric("Predicted virality", predicted_category)
            r2.metric(
                "Model probability for this category",
                f"{predicted_probability:.1%}",
            )

            st.markdown("**Probability for each category**")
            st.dataframe(
                pd.DataFrame({
                    "Virality category": labels,
                    "Model probability": [round(float(p), 4) for p in probabilities],
                }),
                use_container_width=True,
            )

            st.caption(
                "This is a model-estimated category, not a guarantee of "
                "future reach. Categories are relative to the historical "
                "dataset."
            )

        except Exception as exc:
            st.error(f"Could not generate virality prediction: {exc}")
            st.exception(exc)