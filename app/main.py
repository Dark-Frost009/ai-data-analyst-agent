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

import hmac
import os
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

from app.ui.analysis import _run_analysis
from app.config import config
from app.core.agent import (
    AgentCapacityError,
    AgentChartError,
    AgentExecutionError,
    AgentExplanationError,
    AgentPlanningError,
    AgentValidationError,
    DataAnalystAgent,
)
from app.core.data_loader import (
    CSVEncodingError,
    CSVParsingError,
    DataLoaderError,
    EmptyFileError,
    FileTooLargeError,
    load_csv,
)
from app.core.data_profiler import profile_dataframe
from app.core.query_planner import MAX_CONVERSATION_TURNS
from app.utils.logger import get_logger


logger = get_logger(__name__)

ACCESS_PASSWORD_ENVIRONMENT_VARIABLE = "APP_ACCESS_PASSWORD"


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

st.html("<style>" + (PROJECT_ROOT / "app/ui/styles.css").read_text(encoding="utf-8") + "</style>")


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

    if "uploaded_file_signature" not in st.session_state:
        st.session_state.uploaded_file_signature = None

    if "uploader_key_version" not in st.session_state:
        st.session_state.uploader_key_version = 0

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "demo_access_granted" not in st.session_state:
        st.session_state.demo_access_granted = False


def _render_public_demo_access_gate() -> bool:
    """Return whether this session may use an optionally protected demo."""

    expected_password = os.getenv(ACCESS_PASSWORD_ENVIRONMENT_VARIABLE, "")

    # Local development remains frictionless until a deployment explicitly
    # supplies a non-empty secret through its environment or secret manager.
    if not expected_password.strip():
        if config.is_production:
            st.error("The demo is locked until APP_ACCESS_PASSWORD is configured.")
            return False
        return True

    if st.session_state.demo_access_granted:
        return True

    st.title("🔒 Private Portfolio Demo")
    st.caption(
        "This demo is access-controlled to protect the AI service quota "
        "used for analysis."
    )

    access_password = st.text_input(
        "Demo access code",
        type="password",
        key="demo_access_password",
    )

    if st.button(
        "Unlock demo",
        type="primary",
        use_container_width=True,
    ):
        if hmac.compare_digest(access_password, expected_password):
            st.session_state.demo_access_granted = True
            st.session_state.pop("demo_access_password", None)
            logger.info("Portfolio demo access granted")
            st.rerun()
        else:
            logger.warning("Portfolio demo access denied")
            st.error("The access code is incorrect. Please try again.")

    return False


def _uploaded_file_signature(uploaded_file) -> tuple:
    """Return Streamlit upload metadata that changes for a new upload."""
    return (
        getattr(uploaded_file, "file_id", None),
        getattr(uploaded_file, "name", None),
        getattr(uploaded_file, "size", None),
    )


def _clear_dataset() -> None:
    """Clear all analysis state and force Streamlit to create a fresh uploader."""
    st.session_state.dataframe = None
    st.session_state.dataset_profile = None
    st.session_state.agent = None
    st.session_state.last_result = None
    st.session_state.uploaded_filename = None
    st.session_state.uploaded_file_signature = None
    st.session_state.conversation_history = []
    st.session_state.uploader_key_version += 1

    logger.info("Dataset cleared from Streamlit session")


def _clear_conversation_history() -> None:
    """Clear session-local follow-up context without clearing the dataset."""

    st.session_state.conversation_history = []
    logger.info("Conversation context cleared from Streamlit session")


def _load_uploaded_file(
    uploaded_file,
    upload_signature: tuple,
) -> None:
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
        return

    st.session_state.dataframe = dataframe
    st.session_state.dataset_profile = dataset_profile
    st.session_state.agent = agent
    st.session_state.last_result = None
    st.session_state.conversation_history = []

    st.session_state.uploaded_filename = getattr(
        uploaded_file,
        "name",
        "uploaded.csv",
    )
    st.session_state.uploaded_file_signature = upload_signature

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

            if is_horizontal:

                # chart_generator.py already returns ranking data in
                # descending metric order.
                #
                # Plotly horizontal categorical axes are displayed
                # bottom-to-top, so reversing the category array places
                # the highest-ranked category at the top.
                fig = px.bar(
                    chart_df,
                    x=value_col,
                    y=category_col,
                    orientation="h",
                )

                fig.update_layout(
                    yaxis=dict(
                        categoryorder="array",
                        categoryarray=list(
                            reversed(
                                chart_df[category_col].tolist()
                            )
                        ),
                    ),
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis_title=value_col,
                    yaxis_title="",
                )

            else:

                fig = px.bar(
                    chart_df,
                    x=category_col,
                    y=value_col,
                )

                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis_title=category_col,
                    yaxis_title=value_col,
                )

            st.plotly_chart(
                fig,
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
            "⚠️ The analysis succeeded, but the chart could not be rendered."
        )

        return

    # "Other" is a summary bucket, not a dropped tail — make it visible
    # whenever the chart actually gained one (bar, grouped_bar, or pie;
    # line/scatter never set this key, so this is a no-op for them).
    if rendered and chart_spec.get("has_other"):

        other_count = chart_spec.get("other_count", 0)

        st.caption(
            f"ℹ️ Showing the top categories by value — {other_count:,} "
            "additional categor"
            + ("y is" if other_count == 1 else "ies are")
            + ' grouped into "Other".'
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

    for warning in getattr(result, "warnings", []):
        st.warning(f"⚠️ {warning}")

    # ----------------------------------------------------------------------
    # 1. Key insight / explanation
    # ----------------------------------------------------------------------

    if result.explanation:

        st.subheader("💡 Key Insight")

        st.write(result.explanation)

    # ----------------------------------------------------------------------
    # 2. Visualization
    # #    Only rendered when the Visualization Intelligence stage decided
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

        if query_result.truncated:
            st.caption(
                f"Rows returned: {query_result.row_count:,} "
                "| Reached the result limit; additional rows may be "
                "available."
            )
        else:
            st.caption(
                f"Rows returned: {query_result.row_count:,}"
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

    if not _render_public_demo_access_gate():
        return

    logger.info(
        "AI Data Analyst Agent started | env=%s | provider=%s",
        config.app_env,
        config.llm_provider,
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
            key=f"csv_uploader_{st.session_state.uploader_key_version}",
            help=(
                f"Maximum upload size: "
                f"{config.max_upload_size_mb} MB"
            ),
        )

        if uploaded_file is not None:

            current_upload_signature = _uploaded_file_signature(
                uploaded_file
            )

            if (
                current_upload_signature
                != st.session_state.uploaded_file_signature
            ):

                _load_uploaded_file(
                    uploaded_file,
                    upload_signature=current_upload_signature,
                )

        if st.button(
            "Clear active dataset",
            disabled=st.session_state.dataframe is None,
            use_container_width=True,
        ):
            _clear_dataset()
            st.rerun()

        if st.button(
            "Clear follow-up context",
            disabled=not st.session_state.conversation_history,
            use_container_width=True,
            help=(
                "Forget prior questions from this browser session while "
                "keeping the current dataset loaded."
            ),
        ):
            _clear_conversation_history()
            st.rerun()

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
            f"AI provider: `{config.llm_provider}`"
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
                        Groq translate them into SQL.
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
                    and Groq.
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

    history_count = len(st.session_state.conversation_history)

    if history_count:
        st.caption(
            f"💬 Follow-up context is active from the last "
            f"{history_count} successful "
            f"{'question' if history_count == 1 else 'questions'} in "
            "this browser session."
        )

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
            AI Data Analyst Agent · Secure SQL · DuckDB · Groq
        </div>
        """
    )


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    main()