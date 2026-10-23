# Big Data Analysis Mini Project Report

## Title
**Social Media Trend Detection and Sentiment Analytics Using PySpark**

## 1. Aim
To extend the Experiment 1 social media analytics case study by applying PySpark-based data processing, trend analysis, engagement analysis, and machine-learning classification to a multi-platform social media dataset.

## 2. Problem statement
Social media posts and interactions are generated across multiple platforms and contain varied metadata, timestamps, engagement counts and potentially sentiment labels. The project analyzes these data to discover platform/topic patterns and changes over time. This extends the original Experiment 1 case study rather than introducing a different problem.

## 3. Dataset
- Source: [Multi-Platform Social Sentiment Evolution, Kaggle](https://www.kaggle.com/datasets/sohumgokhale/multi-platform-social-sentiment-evolution)
- Local file name: `multi_platform_social_sentiment_evolution.csv`
- Number of rows before cleaning: **[copy from dashboard]**
- Number of rows after deduplication: **[copy from dashboard]**
- Number of columns: **[copy from dashboard]**
- Detected fields: **[copy the detected schema shown in dashboard]**
- Dataset licence/access notes: **[check Kaggle dataset page before redistribution]**

Do not write that the dataset contains views, likes, shares, comments, hashtags, regions, or timestamps unless those columns are present in your downloaded CSV. List unavailable fields as limitations.

## 4. Big Data characteristics
- **Volume:** describe the actual row count and file size from your dataset. Local processing demonstrates the workflow, not cluster-scale throughput.
- **Variety:** identify the numeric, categorical, timestamp, text, and sentiment fields actually present.
- **Velocity:** this project analyzes a static CSV snapshot. Live velocity is not implemented; simulated streaming may be future work.
- **Veracity:** duplicate records are checked, numeric engagement columns are cast where available, and missing categorical values are marked `Unknown`.

## 5. Tools and technologies
- Python
- Apache Spark / PySpark DataFrames
- PySpark MLlib Random Forest classifier (when suitable columns are available)
- Streamlit dashboard
- Pandas for dashboard presentation/export
- Kaggle CSV dataset

This local version does not implement HDFS, YARN, HBase, ZooKeeper, Sqoop, or live platform API ingestion. Those may be discussed as possible future extensions only.

## 6. Methodology
1. Download the CSV from Kaggle.
2. Load the CSV with PySpark.
3. Detect common column aliases and standardize recognized fields.
4. Check duplicate records and handle missing categorical values.
5. Aggregate records by platform, topic/hashtag and sentiment where the columns exist.
6. Compare available engagement metrics by platform/content type/topic.
7. Parse dates and produce daily trend counts when timestamps are available.
8. Train and evaluate a Random Forest classifier if the dataset contains a usable sentiment label or sufficient engagement metrics.
9. Display and export results in Streamlit.

## 7. Results
Fill this section only after running the application.

- Records before deduplication: **[actual value]**
- Records after deduplication: **[actual value]**
- Duplicate records removed: **[actual value]**
- Top platforms/topics/sentiments: **[copy actual dashboard output]**
- Engagement comparison: **[copy actual output if metrics exist]**
- Model task: **[actual task, or state unavailable]**
- Test records: **[actual value if trained]**
- Accuracy: **[actual value if trained]**
- Weighted F1: **[actual value if trained]**

Attach screenshots of the dashboard, platform/topic charts, sentiment chart if available, and model output if training succeeds. Do not insert example or fabricated metrics as project results.

## 8. Discussion
Explain what patterns are visible in your actual charts. Distinguish association from causation. If engagement columns are absent, do not make engagement claims. If sentiment labels are absent, do not claim sentiment classification was performed.

## 9. Limitations
- Findings depend on the columns and quality of the downloaded dataset.
- Dataset is a stored snapshot, not a live stream.
- The project runs on a local Spark session, not a distributed multi-node cluster.
- Random Forest evaluation depends on the available labels and feature columns.
- Synthetic fallback data, if used, must be labelled synthetic and must not be presented as Kaggle data.

## 10. Conclusion
Write a conclusion based on the actual analysis and model results. Explain how PySpark aggregation supports scalable analysis of social media data and how the dashboard makes the findings accessible.

## 11. Future scope
- Add a real streaming source through a permitted API or a message broker such as Kafka.
- Store large raw datasets in HDFS or object storage.
- Compare models and tune hyperparameters.
- Add text sentiment analysis only if text data and suitable labels are available.
- Schedule periodic ingestion and monitor changes in topic or sentiment distributions.
