"""
AI Data Analyst Agent — Streamlit application entry point.

Application pipeline:

    CSV Upload
        ↓
    Data Loading
        ↓
    Dataset Profiling
        ↓
    DataAnalystAgent
        ↓
    Query Planning
        ↓
    SQL Security Validation
        ↓
    DuckDB Execution
        ↓
    Chart Generation
        ↓
    LLM Explanation

The Streamlit layer is responsible for:

- user interaction
- displaying application state
- passing data into the core pipeline
- rendering results

Business logic remains inside app.core.
"""

import sys
from pathlib import Path


# --------------------------------------------------------------------------
# Ensure project root is importable
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------
# Third-party imports
# --------------------------------------------------------------------------

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------------------------------
# Application imports
# --------------------------------------------------------------------------

from app.config import config
from app.core.agent import DataAnalystAgent
from app.core.data_loader import (
    CSVEncodingError,
    CSVParsingError,
    DataLoaderError,
    EmptyFileError,
    FileTooLargeError,
    load_csv,
)
from app.core.data_profiler import profile_dataframe
from app.core.explainer import ExplainerError
from app.core.llm_client import BedrockClientError
from app.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title=config.app_name,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Global CSS
# --------------------------------------------------------------------------

st.html(
    """
    <style>

        /* ================================================================
           GLOBAL APPLICATION
           ================================================================ */

        .stApp {
            background:
                radial-gradient(
                    circle at 82% 5%,
                    rgba(219, 234, 254, 0.55),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 8% 38%,
                    rgba(239, 246, 255, 0.72),
                    transparent 30%
                ),
                #f8fafc;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }


        /* ================================================================
           TYPOGRAPHY
           ================================================================ */

        h1 {
            font-size: 2.65rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.045em !important;
            color: #0f172a !important;
            margin-bottom: 0.4rem !important;
        }

        h2 {
            font-size: 1.55rem !important;
            font-weight: 750 !important;
            letter-spacing: -0.025em !important;
            color: #0f172a !important;
        }

        h3 {
            font-weight: 700 !important;
            color: #0f172a !important;
        }

        p {
            color: #64748b;
        }


        /* ================================================================
           SIDEBAR
           ================================================================ */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    rgba(255, 255, 255, 0.98),
                    rgba(248, 250, 252, 0.98)
                );

            border-right: 1px solid #e2e8f0;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] hr {
            margin: 1.25rem 0;
        }


        /* ================================================================
           APP EYEBROW
           ================================================================ */

        .app-eyebrow {
            display: block;
            width: 100%;

            margin: 0 0 0.65rem 0;
            padding: 0;

            font-size: 0.74rem;
            line-height: 1.2;

            font-weight: 800;

            letter-spacing: 0.20em;

            color: #64748b;

            text-align: left;
        }


        /* ================================================================
           HERO
           ================================================================ */

        .hero-card {
            width: 100%;
            box-sizing: border-box;

            margin-top: 2rem;
            margin-bottom: 2.25rem;

            padding: 3.6rem 3rem;

            border: 1px solid #dbe4ee;
            border-radius: 26px;

            background:
                radial-gradient(
                    circle at 50% 0%,
                    rgba(219, 234, 254, 0.78),
                    transparent 45%
                ),
                radial-gradient(
                    circle at 90% 30%,
                    rgba(239, 246, 255, 0.78),
                    transparent 32%
                ),
                rgba(255, 255, 255, 0.94);

            box-shadow:
                0 25px 65px rgba(15, 23, 42, 0.065),
                0 5px 18px rgba(15, 23, 42, 0.025),
                inset 0 1px 0 rgba(255, 255, 255, 0.95);

            text-align: center;
        }

        .hero-icon {
            width: 82px;
            height: 82px;

            margin: 0 auto 1.35rem auto;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 22px;

            background:
                linear-gradient(
                    145deg,
                    #eff6ff,
                    #dbeafe
                );

            border: 1px solid #bfdbfe;

            font-size: 2.8rem;
            line-height: 1;

            box-shadow:
                0 12px 30px rgba(37, 99, 235, 0.12);
        }

        .hero-eyebrow {
            margin-bottom: 0.65rem;

            font-size: 0.75rem;
            line-height: 1.2;

            font-weight: 800;

            letter-spacing: 0.20em;

            color: #64748b;
        }

        .hero-title {
            margin-bottom: 1rem;

            font-size: 2.35rem;
            line-height: 1.12;

            font-weight: 800;

            letter-spacing: -0.045em;

            color: #0f172a;
        }

        .hero-description {
            max-width: 760px;

            margin: 0 auto;

            font-size: 1rem;
            line-height: 1.75;

            color: #64748b;
        }


        /* ================================================================
           SECTION LABEL
           ================================================================ */

        .section-label {
            margin-bottom: 1rem;

            font-size: 0.76rem;
            font-weight: 800;

            letter-spacing: 0.12em;
            text-transform: uppercase;

            color: #64748b;
        }


        /* ================================================================
           FEATURE CARDS
           ================================================================ */

        .feature-card {
            height: 100%;
            min-height: 185px;

            box-sizing: border-box;

            padding: 1.45rem;

            border: 1px solid #e2e8f0;
            border-radius: 18px;

            background:
                linear-gradient(
                    180deg,
                    rgba(255, 255, 255, 0.96),
                    rgba(248, 250, 252, 0.90)
                );

            box-shadow:
                0 8px 24px rgba(15, 23, 42, 0.035);

            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease,
                border-color 0.18s ease;
        }

        .feature-card:hover {
            transform: translateY(-2px);

            border-color: #cbd5e1;

            box-shadow:
                0 14px 30px rgba(15, 23, 42, 0.065);
        }

        .feature-icon {
            width: 44px;
            height: 44px;

            margin-bottom: 1rem;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 12px;

            background: #eff6ff;
            border: 1px solid #dbeafe;

            font-size: 1.25rem;
        }

        .feature-title {
            margin-bottom: 0.45rem;

            font-size: 1rem;
            font-weight: 750;

            color: #0f172a;
        }

        .feature-text {
            font-size: 0.89rem;
            line-height: 1.6;

            color: #64748b;
        }


        /* ================================================================
           METRIC CARDS
           ================================================================ */

        div[data-testid="stMetric"] {
            min-height: 115px;

            background:
                linear-gradient(
                    145deg,
                    rgba(255, 255, 255, 0.97),
                    rgba(248, 250, 252, 0.91)
                );

            border: 1px solid #e2e8f0;

            border-radius: 17px;

            padding: 1.05rem 1.2rem;

            box-shadow:
                0 7px 22px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 650;
            color: #64748b;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
        }


        /* ================================================================
           FILE UPLOADER
           ================================================================ */

        [data-testid="stFileUploader"] {
            background: transparent;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed #cbd5e1;
            border-radius: 14px;

            background:
                linear-gradient(
                    180deg,
                    #f8fafc,
                    #ffffff
                );

            transition:
                border-color 0.18s ease,
                background 0.18s ease;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #93c5fd;
            background: #f8fbff;
        }


        /* ================================================================
           TEXT AREA
           ================================================================ */

        div[data-testid="stTextArea"] textarea {
            min-height: 110px;

            border-radius: 15px;

            border: 1px solid #cbd5e1;

            padding: 1rem 1.05rem;

            background: #ffffff;

            font-size: 1rem;

            color: #0f172a;

            box-shadow:
                0 3px 12px rgba(15, 23, 42, 0.025);
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #93c5fd;

            box-shadow:
                0 0 0 3px rgba(59, 130, 246, 0.10),
                0 5px 16px rgba(15, 23, 42, 0.035);
        }


        /* ================================================================
           SELECT BOX
           ================================================================ */

        div[data-testid="stSelectbox"] > div > div {
            border-radius: 12px;
            border-color: #cbd5e1;
        }


        /* ================================================================
           PRIMARY BUTTON
           ================================================================ */

        div.stButton > button[kind="primary"] {
            width: 100%;

            min-height: 3.15rem;

            border-radius: 13px;

            font-weight: 750;

            letter-spacing: 0.01em;

            border: none;

            box-shadow:
                0 8px 20px rgba(37, 99, 235, 0.18);

            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);

            box-shadow:
                0 11px 26px rgba(37, 99, 235, 0.25);
        }


        /* ================================================================
           DATAFRAMES
           ================================================================ */

        div[data-testid="stDataFrame"] {
            border-radius: 15px;
            overflow: hidden;

            border: 1px solid #e2e8f0;

            box-shadow:
                0 5px 18px rgba(15, 23, 42, 0.025);
        }


        /* ================================================================
           CODE BLOCK
           ================================================================ */

        div[data-testid="stCode"] {
            border-radius: 15px;
            overflow: hidden;

            border: 1px solid #e2e8f0;

            box-shadow:
                0 5px 18px rgba(15, 23, 42, 0.025);
        }


        /* ================================================================
           EXPANDERS
           ================================================================ */

        div[data-testid="stExpander"] {
            border: 1px solid #e2e8f0;
            border-radius: 15px;

            background:
                rgba(255, 255, 255, 0.78);

            overflow: hidden;
        }


        /* ================================================================
           ALERTS
           ================================================================ */

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }


        /* ================================================================
           RESULT HEADER
           ================================================================ */

        .result-header {
            margin-bottom: 0.65rem;

            font-size: 0.76rem;
            font-weight: 800;

            letter-spacing: 0.13em;
            text-transform: uppercase;

            color: #64748b;
        }


        /* ================================================================
           INFO STRIP
           ================================================================ */

        .info-strip {
            display: flex;
            align-items: center;
            gap: 0.65rem;

            margin-top: 1rem;

            padding: 0.95rem 1.1rem;

            border: 1px solid #dbeafe;
            border-radius: 13px;

            background:
                linear-gradient(
                    90deg,
                    rgba(239, 246, 255, 0.95),
                    rgba(248, 250, 252, 0.85)
                );

            color: #475569;

            font-size: 0.88rem;
        }


        /* ================================================================
           FOOTER
           ================================================================ */

        .app-footer {
            margin-top: 3rem;
            padding-top: 1.5rem;

            border-top: 1px solid #e2e8f0;

            text-align: center;

            font-size: 0.78rem;

            color: #94a3b8;
        }


        /* ================================================================
           RESPONSIVE
           ================================================================ */

        @media (max-width: 900px) {

            .block-container {
                padding-left: 1.2rem;
                padding-right: 1.2rem;
            }

            .hero-card {
                margin-top: 1.5rem;
                padding: 2.5rem 1.4rem;
            }

            .hero-title {
                font-size: 1.85rem;
            }

            .hero-description {
                font-size: 0.94rem;
            }

            h1 {
                font-size: 2.15rem !important;
            }
        }

    </style>
    """
)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def _initialize_session_state() -> None:
    """Initialize Streamlit session-state values."""

    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None

    if "dataset_profile" not in st.session_state:
        st.session_state.dataset_profile = None

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None


def _load_uploaded_file(uploaded_file) -> None:
    """
    Load and profile an uploaded CSV file.

    Session state is updated only after the complete pipeline succeeds.
    """

    try:
        dataframe = load_csv(uploaded_file)

        dataset_profile = profile_dataframe(dataframe)

        agent = DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile=dataset_profile,
            max_result_rows=config.max_query_result_rows,
        )

    except FileTooLargeError as exc:
        st.error(f"📦 File is too large: {exc}")
        return

    except EmptyFileError as exc:
        st.error(f"📭 Empty file: {exc}")
        return

    except CSVEncodingError as exc:
        st.error(f"🔤 CSV encoding error: {exc}")
        return

    except CSVParsingError as exc:
        st.error(f"📄 CSV parsing error: {exc}")
        return

    except DataLoaderError as exc:
        st.error(f"❌ Could not load the CSV file: {exc}")
        return

    except Exception as exc:
        logger.exception(
            "Unexpected error while preparing uploaded dataset"
        )

        st.error(
            "❌ An unexpected error occurred while preparing the dataset."
        )

        st.exception(exc)
        return

    st.session_state.dataframe = dataframe
    st.session_state.dataset_profile = dataset_profile
    st.session_state.agent = agent
    st.session_state.last_result = None

    st.session_state.uploaded_filename = getattr(
        uploaded_file,
        "name",
        "uploaded.csv",
    )

    logger.info(
        "Dataset loaded into Streamlit session | filename=%s | "
        "rows=%d | columns=%d",
        st.session_state.uploaded_filename,
        len(dataframe),
        len(dataframe.columns),
    )

    st.success(
        f"✅ Successfully loaded "
        f"**{st.session_state.uploaded_filename}** "
        f"({len(dataframe):,} rows × {len(dataframe.columns)} columns)."
    )


def _display_dataset_overview() -> None:
    """Display basic information about the loaded dataset."""

    dataframe = st.session_state.dataframe
    profile = st.session_state.dataset_profile

    if dataframe is None or profile is None:
        return

    st.html(
        """
        <div class="section-label">
            DATASET OVERVIEW
        </div>
        """
    )

    st.subheader("📋 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{profile.row_count:,}",
        )

    with col2:
        st.metric(
            "Columns",
            f"{profile.column_count:,}",
        )

    with col3:
        total_missing = sum(
            column.null_count
            for column in profile.columns
        )

        st.metric(
            "Missing Values",
            f"{total_missing:,}",
        )

    with st.expander("👀 Preview data", expanded=False):
        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("🔎 Dataset profile", expanded=False):

        profile_data = [
            {
                "Column": column.name,
                "Pandas Type": column.pandas_dtype,
                "Inferred Type": column.inferred_type,
                "Null Count": column.null_count,
                "Null %": round(column.null_percentage, 2),
                "Unique Values": column.unique_count,
            }
            for column in profile.columns
        ]

        st.dataframe(
            profile_data,
            use_container_width=True,
            hide_index=True,
        )


def _display_chart(chart_spec) -> None:
    """
    Render a chart specification returned by the agent's Visualization
    Intelligence stage, or a subtle "not needed" notice when the engine
    decided a chart wouldn't meaningfully add anything for this result.

    `chart_spec` is either None (no chart was useful for this result) or
    a JSON-safe dict produced by app.core.chart_generator.generate_chart.
    """

    st.html(
        """
        <div class="result-header">
            VISUALIZATION
        </div>
        """
    )

    data = (chart_spec or {}).get("data", [])

    if not chart_spec or not data:
        st.caption(
            "📉 Visualization isn't necessary for this result — the "
            "table below communicates it clearly."
        )
        return

    chart_type = chart_spec.get("chart_type")
    rendered = False

    try:

        if chart_type == "bar":

            category_col = chart_spec.get("x_column", "category")
            value_col = chart_spec.get("y_column", "value")
            is_horizontal = chart_spec.get("orientation") == "horizontal"

            chart_df = pd.DataFrame(
                {
                    category_col: [item["category"] for item in data],
                    value_col: [item["value"] for item in data],
                }
            )

            st.bar_chart(
                chart_df,
                x=category_col,
                y=value_col,
                horizontal=is_horizontal,
                sort=f"-{value_col}",
                use_container_width=True,
            )

            rendered = True

        elif chart_type == "grouped_bar":

            _display_grouped_bar_chart(chart_spec, data)

            rendered = True

        elif chart_type == "line":

            y_columns = chart_spec.get("y_columns")

            if y_columns and len(y_columns) > 1:

                # Multi-series time chart: every data point already
                # carries "x" plus one key per numeric series.
                chart_df = pd.DataFrame(data).set_index("x")

                st.line_chart(
                    chart_df,
                    y=y_columns,
                    use_container_width=True,
                )

            else:

                chart_df = pd.DataFrame(
                    {
                        "x": [item["x"] for item in data],
                        "y": [item["y"] for item in data],
                    }
                ).set_index("x")

                st.line_chart(
                    chart_df,
                    use_container_width=True,
                )

            rendered = True

        elif chart_type == "scatter":

            chart_df = pd.DataFrame(
                {
                    "x": [item["x"] for item in data],
                    "y": [item["y"] for item in data],
                }
            )

            st.scatter_chart(
                chart_df,
                x="x",
                y="y",
                use_container_width=True,
            )

            rendered = True

        elif chart_type == "pie":

            _display_pie_chart(data)

            rendered = True

        else:

            logger.warning(
                "Unknown chart type returned by chart generator: %s",
                chart_type,
            )

    except Exception as exc:

        logger.exception("Failed to render chart")

        st.warning(
            "⚠️ The analysis succeeded, but the chart "
            f"could not be rendered: {exc}"
        )

        return

    # "Other" is a summary bucket, not a dropped tail — make it visible
    # whenever the chart actually gained one (bar, grouped_bar, or pie;
    # line/scatter never set this key, so this is a no-op for them).
    if rendered and chart_spec.get("has_other"):

        other_count = chart_spec.get("other_count", 0)

        st.caption(
            f"ℹ️ Showing the top categories by value — {other_count:,} "
            "additional categor" + ("y is" if other_count == 1 else "ies are") +
            " grouped into \"Other\"."
        )


def _display_grouped_bar_chart(chart_spec: dict, data: list) -> None:
    """
    Render a grouped (multi-series) bar chart via Plotly: one categorical
    dimension on the x-axis, with every requested numeric metric drawn
    as its own series of bars within each category's group.
    """

    category_col = chart_spec.get("x_column", "category")
    value_columns = chart_spec.get("y_columns", [])

    chart_df = pd.DataFrame(
        {
            category_col: [item["category"] for item in data],
            **{
                column: [item[column] for item in data]
                for column in value_columns
            },
        }
    )

    fig = px.bar(
        chart_df,
        x=category_col,
        y=value_columns,
        barmode="group",
    )

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        legend_title_text="",
        xaxis_title=category_col,
        yaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def _display_pie_chart(data: list) -> None:
    """Render a pie/donut chart from chart_generator's pie data points."""

    labels = [item["label"] for item in data]
    values = [item["value"] for item in data]

    fig = px.pie(
        names=labels,
        values=values,
        hole=0.45,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def _run_analysis(
    question: str,
    chart_type: str | None,
) -> None:
    """Run the complete DataAnalystAgent pipeline."""

    agent = st.session_state.agent

    if agent is None:
        st.error(
            "Please upload a CSV file before asking a question."
        )
        return

    with st.spinner("🤖 Analyzing your question..."):

        try:

            result = agent.run(
                question=question,
                generate_explanation=True,
                generate_chart_spec=True,
                chart_type=chart_type,
            )

        except BedrockClientError as exc:

            logger.exception("Bedrock request failed")

            st.error(
                "❌ The AI model could not process the request."
            )

            st.exception(exc)
            return

        except ExplainerError as exc:

            logger.exception(
                "Explanation generation failed"
            )

            st.error(
                "⚠️ The SQL analysis completed, but generating "
                "the explanation failed."
            )

            st.exception(exc)
            return

        except Exception as exc:

            logger.exception(
                "Unexpected agent execution error"
            )

            st.error(
                "❌ The analysis could not be completed."
            )

            st.exception(exc)
            return

    st.session_state.last_result = result


def _display_analysis_result() -> None:
    """Display the latest analysis result."""

    result = st.session_state.last_result

    if result is None:
        return

    st.divider()

    st.html(
        """
        <div class="section-label">
            ANALYSIS RESULT
        </div>
        """
    )

    # ----------------------------------------------------------------------
    # 1. Key insight / explanation
    # ----------------------------------------------------------------------

    if result.explanation:

        st.subheader("💡 Key Insight")

        st.write(result.explanation)

    # ----------------------------------------------------------------------
    # 2. Visualization
    #    Only rendered when the Visualization Intelligence stage decided
    #    it adds value; otherwise a subtle notice is shown in its place
    #    (see _display_chart), never a silent empty gap.
    # ----------------------------------------------------------------------

    _display_chart(result.chart)

    # ----------------------------------------------------------------------
    # 3. Query result
    # ----------------------------------------------------------------------

    st.subheader("📊 Query Result")

    query_result = result.query_result

    if query_result is not None:

        st.caption(
            f"Rows returned: {query_result.row_count:,} "
            f"| Truncated: {query_result.truncated}"
        )

        st.dataframe(
            query_result.dataframe,
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------------------------------------------------
    # 4. Generated SQL
    #    Collapsed by default so it supports the result instead of
    #    dominating it.
    # ----------------------------------------------------------------------

    with st.expander("🧾 Generated SQL", expanded=False):

        st.code(
            result.sql,
            language="sql",
        )


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------

def main() -> None:
    """Run the Streamlit application."""

    _initialize_session_state()

    logger.info(
        "AI Data Analyst Agent started | env=%s | region=%s",
        config.app_env,
        config.aws_region,
    )

    # ----------------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------------

    st.html(
        """
        <div class="app-eyebrow">
            INTELLIGENT DATA ANALYSIS
        </div>
        """
    )

    st.title(f"📊 {config.app_name}")

    st.caption(
        "Upload a CSV, ask a question in plain English, and let the "
        "AI Data Analyst Agent plan, validate, execute, explain, and "
        "visualize the result."
    )

    # ----------------------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------------------

    with st.sidebar:

        st.header("⚙️ Dataset")

        st.caption(
            "Upload a CSV dataset to activate the AI analyst."
        )

        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help=(
                f"Maximum upload size: "
                f"{config.max_upload_size_mb} MB"
            ),
        )

        if uploaded_file is not None:

            current_filename = getattr(
                uploaded_file,
                "name",
                None,
            )

            if current_filename != st.session_state.uploaded_filename:

                _load_uploaded_file(uploaded_file)

        st.divider()

        st.header("📈 Visualization")

        chart_option = st.selectbox(
            "Chart type",
            options=[
                "Automatic",
                "Bar",
                "Grouped Bar",
                "Line",
                "Pie",
                "Scatter",
            ],
            index=0,
        )

        chart_type_map = {
            "Automatic": None,
            "Bar": "bar",
            "Grouped Bar": "grouped_bar",
            "Line": "line",
            "Pie": "pie",
            "Scatter": "scatter",
        }

        selected_chart_type = chart_type_map[chart_option]

        st.divider()

        st.caption(
            f"Environment: `{config.app_env}`"
        )

        st.caption(
            f"Region: `{config.aws_region}`"
        )

    # ----------------------------------------------------------------------
    # Empty state
    # ----------------------------------------------------------------------

    if st.session_state.dataframe is None:

        st.info(
            "👈 Upload a CSV file from the sidebar to begin."
        )

        # ------------------------------------------------------------------
        # Premium hero
        # ------------------------------------------------------------------

        st.html(
            """
            <div class="hero-card">

                <div class="hero-icon">
                    📊
                </div>

                <div class="hero-eyebrow">
                    AI-POWERED DATA ANALYTICS
                </div>

                <div class="hero-title">
                    Talk to your data.
                </div>

                <div class="hero-description">
                    Upload a CSV dataset and ask questions in
                    plain English. The AI Data Analyst Agent
                    transforms your questions into secure SQL,
                    analyzes the results, and turns them into
                    clear insights and visualizations.
                </div>

            </div>
            """
        )

        # ------------------------------------------------------------------
        # Features
        # ------------------------------------------------------------------

        st.html(
            """
            <div class="section-label">
                WHAT THIS APP CAN DO
            </div>
            """
        )

        feature_col1, feature_col2, feature_col3 = st.columns(3)

        with feature_col1:

            st.html(
                """
                <div class="feature-card">

                    <div class="feature-icon">
                        📂
                    </div>

                    <div class="feature-title">
                        Understand your data
                    </div>

                    <div class="feature-text">
                        Load CSV datasets, profile columns,
                        inspect data types, and preview your data.
                    </div>

                </div>
                """
            )

        with feature_col2:

            st.html(
                """
                <div class="feature-card">

                    <div class="feature-icon">
                        🤖
                    </div>

                    <div class="feature-title">
                        Ask questions naturally
                    </div>

                    <div class="feature-text">
                        Ask questions in plain English and let
                        Amazon Bedrock translate them into SQL.
                    </div>

                </div>
                """
            )

        with feature_col3:

            st.html(
                """
                <div class="feature-card">

                    <div class="feature-icon">
                        🛡️
                    </div>

                    <div class="feature-title">
                        Analyze safely
                    </div>

                    <div class="feature-text">
                        Validate generated SQL, execute safe
                        queries through DuckDB, and return results.
                    </div>

                </div>
                """
            )

        # ------------------------------------------------------------------
        # Technology strip
        # ------------------------------------------------------------------

        st.html(
            """
            <div class="info-strip">
                <span>✨</span>
                <span>
                    Powered by secure SQL validation, DuckDB,
                    and Amazon Bedrock.
                </span>
            </div>
            """
        )

        # ------------------------------------------------------------------
        # Footer
        # ------------------------------------------------------------------

        st.html(
            """
            <div class="app-footer">
                💡 Upload your dataset from the sidebar to start
                asking questions.
            </div>
            """
        )

        return

    # ----------------------------------------------------------------------
    # Dataset overview
    # ----------------------------------------------------------------------

    _display_dataset_overview()

    st.divider()

    # ----------------------------------------------------------------------
    # Natural-language question
    # ----------------------------------------------------------------------

    st.html(
        """
        <div class="section-label">
            NATURAL LANGUAGE QUERY
        </div>
        """
    )

    st.subheader("💬 Ask a question about your data")

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What are the top 10 products by total sales?"
        ),
        height=110,
        label_visibility="collapsed",
    )

    analyze_clicked = st.button(
        "🚀 Analyze",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:

        if not question.strip():

            st.warning(
                "Please enter a question before clicking Analyze."
            )

        else:

            _run_analysis(
                question=question.strip(),
                chart_type=selected_chart_type,
            )

    # ----------------------------------------------------------------------
    # Latest result
    # ----------------------------------------------------------------------

    _display_analysis_result()

    # ----------------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------------

    st.html(
        """
        <div class="app-footer">
            AI Data Analyst Agent · Secure SQL · DuckDB · Amazon Bedrock
        </div>
        """
    )


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    main()