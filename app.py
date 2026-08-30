import os
import glob
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrendSense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# LOAD ALL CSV FILES FROM DATA FOLDER
# ============================================================

@st.cache_data
def load_csv_files():

    files = glob.glob(
        os.path.join(DATA_DIR, "*.csv")
    )

    datasets = {}

    for file in files:

        try:

            df = pd.read_csv(file)

            filename = os.path.basename(file)

            datasets[filename] = df

        except Exception:
            pass

    return datasets


datasets = load_csv_files()


if not datasets:

    st.error(
        "No CSV files were found inside the data folder."
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):

    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:

        key = name.lower()

        if key in column_map:
            return column_map[key]

    return None


def find_dataset_with_column(possible_names):

    for filename, df in datasets.items():

        column = find_column(
            df,
            possible_names
        )

        if column is not None:

            return filename, df, column

    return None, None, None


def to_numeric(df, column):

    if column is None:
        return pd.Series(dtype="float64")

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# FIND REDDIT DATASET
# ============================================================

reddit_df = None
reddit_filename = None


preferred_reddit_files = [
    "reddit_cleaned.csv",
    "reddit_final_analysis.csv",
    "reddit_with_topics.csv"
]


for filename in preferred_reddit_files:

    if filename in datasets:

        reddit_df = datasets[filename]
        reddit_filename = filename

        break


# Fallback: find a dataset containing communityName

if reddit_df is None:

    for filename, df in datasets.items():

        if find_column(
            df,
            [
                "communityName",
                "subreddit",
                "community"
            ]
        ) is not None:

            reddit_df = df
            reddit_filename = filename

            break


# If still not found, use first dataset

if reddit_df is None:

    reddit_filename = list(
        datasets.keys()
    )[0]

    reddit_df = datasets[
        reddit_filename
    ]


# ============================================================
# FIND TREND DATASET
# ============================================================

trend_df = None
trend_filename = None


preferred_trend_files = [
    "final_trending_aggression_results.csv",
    "trending_topics.csv"
]


for filename in preferred_trend_files:

    if filename in datasets:

        trend_df = datasets[filename]
        trend_filename = filename

        break


# Fallback: find dataset containing trend_score

if trend_df is None:

    for filename, df in datasets.items():

        if find_column(
            df,
            [
                "trend_score"
            ]
        ) is not None:

            trend_df = df
            trend_filename = filename

            break


# Fallback to Reddit dataset

if trend_df is None:

    trend_df = reddit_df.copy()
    trend_filename = reddit_filename


# ============================================================
# FIND COLUMNS
# ============================================================

text_col = find_column(
    reddit_df,
    [
        "clean_text",
        "text"
    ]
)


community_col = find_column(
    reddit_df,
    [
        "communityName",
        "community",
        "subreddit"
    ]
)


datetime_col = find_column(
    reddit_df,
    [
        "datetime",
        "date"
    ]
)


toxicity_col = find_column(
    reddit_df,
    [
        "toxicity",
        "toxic",
        "toxicity_score"
    ]
)


aggression_col = find_column(
    reddit_df,
    [
        "aggression_level",
        "aggression",
        "aggression_score"
    ]
)


topic_name_col = find_column(
    trend_df,
    [
        "topic_name",
        "topic",
        "Name",
        "name"
    ]
)


topic_id_col = find_column(
    trend_df,
    [
        "topic_id",
        "Topic"
    ]
)


mention_col = find_column(
    trend_df,
    [
        "mention_count",
        "count",
        "Count"
    ]
)


growth_col = find_column(
    trend_df,
    [
        "growth",
        "growth_score"
    ]
)


trend_score_col = find_column(
    trend_df,
    [
        "trend_score",
        "Trend_Score"
    ]
)


# ============================================================
# CREATE DATA-DRIVEN TREND CLASSIFICATION
# ============================================================

trend_working = trend_df.copy()


if trend_score_col:

    trend_working["_trend_numeric"] = pd.to_numeric(
        trend_working[trend_score_col],
        errors="coerce"
    )


    valid_scores = (
        trend_working["_trend_numeric"]
        .dropna()
    )


    if not valid_scores.empty:

        low_boundary = valid_scores.quantile(
            0.33
        )

        high_boundary = valid_scores.quantile(
            0.67
        )


        def classify_trend(score):

            if pd.isna(score):
                return "Unknown"

            if score <= low_boundary:
                return "Low"

            elif score <= high_boundary:
                return "Medium"

            else:
                return "High"


        trend_working["trend_level"] = (
            trend_working["_trend_numeric"]
            .apply(classify_trend)
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 TrendSense AI")

    st.caption(
        "Social Media Trend Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔥 Trending Topics",
            "⚠️ Aggression Analysis",
            "🌐 Platform Comparison",
            "📁 Data Explorer"
        ]
    )

    st.divider()

    st.write(
        "**Connected data**"
    )

    st.success(
        "🔴 Reddit"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "📊 TrendSense AI"
    )

    st.write(
        "AI-powered social media trend, topic, "
        "toxicity and aggression analysis."
    )

    st.divider()


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_discussions = len(
        reddit_df
    )

    total_topics = len(
        trend_working
    )


    if trend_score_col:

        trend_values = (
            trend_working["_trend_numeric"]
            .dropna()
        )

        highest_trend = (
            trend_values.max()
            if not trend_values.empty
            else 0
        )

    else:

        highest_trend = 0


    if toxicity_col:

        toxicity_values = to_numeric(
            reddit_df,
            toxicity_col
        ).dropna()

        highest_toxicity = (
            toxicity_values.max()
            if not toxicity_values.empty
            else 0
        )

    else:

        highest_toxicity = 0


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Discussions",
            f"{total_discussions:,}"
        )


    with c2:

        st.metric(
            "Topics",
            f"{total_topics:,}"
        )


    with c3:

        st.metric(
            "Highest Trend",
            f"{highest_trend:.2f}"
        )


    with c4:

        st.metric(
            "Highest Toxicity",
            f"{highest_toxicity:.2f}"
        )


    st.divider()


    # --------------------------------------------------------
    # TOP TRENDING TOPICS
    # --------------------------------------------------------

    st.subheader(
        "🔥 Top Trending Topics"
    )


    if (
        topic_name_col
        and trend_score_col
    ):

        top_topics = (
            trend_working
            .sort_values(
                "_trend_numeric",
                ascending=False
            )
            .head(10)
            .copy()
        )


        chart_data = top_topics[
            [
                topic_name_col,
                "_trend_numeric"
            ]
        ].copy()


        chart_data = chart_data.rename(
            columns={
                "_trend_numeric":
                "Trend Score"
            }
        )


        chart_data = chart_data.set_index(
            topic_name_col
        )


        st.bar_chart(
            chart_data,
            use_container_width=True
        )


        # ----------------------------------------------------
        # TREND LEVEL LEGEND
        # ----------------------------------------------------

        st.write(
            "### Trend Level"
        )


        l1, l2, l3 = st.columns(3)


        with l1:

            st.success(
                "🟢 LOW"
            )


        with l2:

            st.warning(
                "🟡 MEDIUM"
            )


        with l3:

            st.error(
                "🔴 HIGH"
            )


# ============================================================
# TRENDING TOPICS
# ============================================================

elif page == "🔥 Trending Topics":

    st.title(
        "🔥 Trending Topics"
    )

    st.write(
        "Topics ranked using the trend scores generated "
        "from the available project data."
    )

    st.divider()


    # --------------------------------------------------------
    # TREND LEVEL EXPLANATION
    # --------------------------------------------------------

    st.info(
        "Trend levels are calculated from the actual trend-score "
        "distribution in your dataset: bottom 33% = Low, "
        "middle 34% = Medium, top 33% = High."
    )


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_text = st.text_input(
        "🔎 Search topics",
        placeholder="Search a topic..."
    )


    topics = trend_working.copy()


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    if (
        search_text
        and
        topic_name_col
    ):

        topics = topics[
            topics[topic_name_col]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        ]


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    if trend_score_col:

        topics = topics.sort_values(
            "_trend_numeric",
            ascending=False
        )


    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    st.subheader(
        "📈 Top 10 Trending Topics"
    )


    if (
        topic_name_col
        and
        trend_score_col
        and
        not topics.empty
    ):

        graph_df = topics[
            [
                topic_name_col,
                "_trend_numeric"
            ]
        ].head(10).copy()


        graph_df = graph_df.rename(
            columns={
                "_trend_numeric":
                "Trend Score"
            }
        )


        graph_df = graph_df.set_index(
            topic_name_col
        )


        st.bar_chart(
            graph_df,
            use_container_width=True
        )


    else:

        st.warning(
            "Trend score/topic columns are not available."
        )


    # --------------------------------------------------------
    # TREND LEVEL COUNTS
    # --------------------------------------------------------

    if "trend_level" in topics.columns:

        st.subheader(
            "📊 Trend Level Distribution"
        )


        level_counts = (
            topics["trend_level"]
            .value_counts()
            .reindex(
                [
                    "Low",
                    "Medium",
                    "High"
                ],
                fill_value=0
            )
        )


        a, b, c = st.columns(3)


        with a:

            st.success(
                f"🟢 Low\n\n"
                f"{level_counts['Low']}"
            )


        with b:

            st.warning(
                f"🟡 Medium\n\n"
                f"{level_counts['Medium']}"
            )


        with c:

            st.error(
                f"🔴 High\n\n"
                f"{level_counts['High']}"
            )


    # --------------------------------------------------------
    # TOPIC TABLE
    # --------------------------------------------------------

    st.subheader(
        "📋 Topic Ranking"
    )


    display_cols = []


    for col in [
        topic_id_col,
        topic_name_col,
        mention_col,
        growth_col,
        trend_score_col
    ]:

        if (
            col
            and
            col not in display_cols
        ):

            display_cols.append(col)


    if "trend_level" in topics.columns:

        display_cols.append(
            "trend_level"
        )


    if display_cols:

        display_df = topics[
            display_cols
        ].head(50).copy()


        # Rename only for cleaner display

        if "trend_level" in display_df.columns:

            display_df = display_df.rename(
                columns={
                    "trend_level":
                    "Trend Level"
                }
            )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            topics.head(50),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# AGGRESSION ANALYSIS
# ============================================================

elif page == "⚠️ Aggression Analysis":

    st.title(
        "⚠️ Aggression & Toxicity Analysis"
    )

    st.write(
        "Analysis based only on the aggression and toxicity "
        "values available in the project datasets."
    )

    st.divider()


    # --------------------------------------------------------
    # FIND AGGRESSION DATASET
    # --------------------------------------------------------

    aggression_file, aggression_df, aggression_col_found = (
        find_dataset_with_column(
            [
                "aggression_level",
                "aggression",
                "aggression_score"
            ]
        )
    )


    # --------------------------------------------------------
    # AGGRESSION
    # --------------------------------------------------------

    if aggression_df is not None:

        st.subheader(
            "⚠️ Aggression Distribution"
        )


        aggression_values = (
            aggression_df[
                aggression_col_found
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )


        unique_values = (
            aggression_values.unique()
        )


        # Categorical aggression

        if len(unique_values) <= 10:

            counts = (
                aggression_values
                .value_counts()
            )


            a, b, c = st.columns(3)


            with a:

                st.metric(
                    "Low",
                    int(
                        counts.get(
                            "Low",
                            0
                        )
                    )
                )


            with b:

                st.metric(
                    "Medium",
                    int(
                        counts.get(
                            "Medium",
                            0
                        )
                    )
                )


            with c:

                st.metric(
                    "High",
                    int(
                        counts.get(
                            "High",
                            0
                        )
                    )
                )


            st.bar_chart(
                counts,
                use_container_width=True
            )


            st.write(
                f"Data source: `{aggression_file}`"
            )


        else:

            numeric_values = pd.to_numeric(
                aggression_df[
                    aggression_col_found
                ],
                errors="coerce"
            ).dropna()


            if not numeric_values.empty:

                st.metric(
                    "Maximum Aggression Score",
                    f"{numeric_values.max():.2f}"
                )


                st.bar_chart(
                    numeric_values.head(100),
                    use_container_width=True
                )


    else:

        st.warning(
            "No aggression column was found in the CSV files "
            "inside the data folder."
        )


    # --------------------------------------------------------
    # TOXICITY
    # --------------------------------------------------------

    toxicity_file, toxicity_df, toxicity_col_found = (
        find_dataset_with_column(
            [
                "toxicity",
                "toxic",
                "toxicity_score"
            ]
        )
    )


    if toxicity_df is not None:

        st.divider()

        st.subheader(
            "🚨 Highest Toxicity Discussions"
        )


        toxic_data = toxicity_df.copy()


        toxic_data["_toxicity_numeric"] = pd.to_numeric(
            toxic_data[
                toxicity_col_found
            ],
            errors="coerce"
        )


        toxic_data = toxic_data.dropna(
            subset=[
                "_toxicity_numeric"
            ]
        )


        toxic_data = toxic_data.sort_values(
            "_toxicity_numeric",
            ascending=False
        )


        text_found = find_column(
            toxic_data,
            [
                "clean_text",
                "text"
            ]
        )


        community_found = find_column(
            toxic_data,
            [
                "communityName",
                "community",
                "subreddit"
            ]
        )


        date_found = find_column(
            toxic_data,
            [
                "datetime",
                "date"
            ]
        )


        toxicity_display = []


        if text_found:

            toxicity_display.append(
                text_found
            )


        toxicity_display.append(
            "_toxicity_numeric"
        )


        if aggression_col_found:

            if aggression_col_found in toxic_data.columns:

                toxicity_display.append(
                    aggression_col_found
                )


        if community_found:

            toxicity_display.append(
                community_found
            )


        if date_found:

            toxicity_display.append(
                date_found
            )


        toxicity_table = toxic_data[
            toxicity_display
        ].head(15).copy()


        toxicity_table = toxicity_table.rename(
            columns={
                "_toxicity_numeric":
                "Toxicity Score"
            }
        )


        st.dataframe(
            toxicity_table,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "No toxicity column was found in the CSV files "
            "inside the data folder."
        )


# ============================================================
# PLATFORM COMPARISON
# ============================================================

elif page == "🌐 Platform Comparison":

    st.title(
        "🌐 Platform Comparison"
    )

    st.write(
        "Current comparison based only on platforms for which "
        "actual project data is available."
    )

    st.divider()


    st.info(
        "No values are fabricated for platforms that are not "
        "present in the project's data folder."
    )


    # --------------------------------------------------------
    # DETECT PLATFORM DATA
    # --------------------------------------------------------

    platform_rows = []


    # Reddit is detected from the dataset

    if reddit_df is not None:

        platform_rows.append(
            {
                "Platform": "Reddit",
                "Status": "Connected",
                "Records": len(reddit_df),
                "Topics": len(trend_working),
                "Data": "Available"
            }
        )


    # Check whether other platform-specific datasets
    # actually exist in the data folder

    platform_keywords = {
        "Instagram": "instagram",
        "TikTok": "tiktok",
        "YouTube": "youtube",
        "X": "twitter"
    }


    filenames_lower = [
        filename.lower()
        for filename in datasets.keys()
    ]


    for platform, keyword in platform_keywords.items():

        matching_files = [
            filename
            for filename in datasets.keys()
            if keyword in filename.lower()
        ]


        if matching_files:

            platform_df = datasets[
                matching_files[0]
            ]


            platform_rows.append(
                {
                    "Platform": platform,
                    "Status": "Connected",
                    "Records": len(platform_df),
                    "Topics": "Available if processed",
                    "Data": matching_files[0]
                }
            )


    # --------------------------------------------------------
    # SHOW ACTUAL DATA
    # --------------------------------------------------------

    if platform_rows:

        platform_df = pd.DataFrame(
            platform_rows
        )


        st.subheader(
            "📊 Available Platform Data"
        )


        st.dataframe(
            platform_df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "No platform datasets were detected."
        )


    st.divider()


    st.subheader(
        "Platform Integration Status"
    )


    status_rows = []


    for platform in [
        "Reddit",
        "Instagram",
        "TikTok",
        "YouTube",
        "X"
    ]:

        matching = [
            filename
            for filename in datasets.keys()
            if platform.lower()
            in filename.lower()
        ]


        if platform == "Reddit":

            status = "Connected"

        elif matching:

            status = "Dataset detected"

        else:

            status = "No dataset detected"


        status_rows.append(
            {
                "Platform": platform,
                "Status": status
            }
        )


    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "📁 Data Explorer":

    st.title(
        "📁 Data Explorer"
    )

    st.write(
        "All CSV files currently available in your data folder."
    )

    st.divider()


    selected_file = st.selectbox(
        "Select CSV dataset",
        sorted(
            datasets.keys()
        )
    )


    selected_df = datasets[
        selected_file
    ]


    c1, c2 = st.columns(2)


    with c1:

        st.metric(
            "Rows",
            f"{len(selected_df):,}"
        )


    with c2:

        st.metric(
            "Columns",
            len(selected_df.columns)
        )


    st.write(
        f"**File:** `{selected_file}`"
    )


    st.dataframe(
        selected_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TrendSense AI • Social Media Trend Intelligence"
)