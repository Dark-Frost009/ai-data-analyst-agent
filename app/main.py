"""
AI Data Analyst Agent — Streamlit application entry point.

This module connects the completed application pipeline:

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

The Streamlit layer is responsible only for:
- user interaction,
- displaying application state,
- passing data into the core pipeline,
- rendering results.

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

import streamlit as st


# --------------------------------------------------------------------------
# Application imports
# --------------------------------------------------------------------------

from app.config import config
from app.core.agent import DataAnalystAgent
from app.core.chart_generator import generate_chart
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
)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------


def _initialize_session_state() -> None:
    """Initialize Streamlit session-state values used by the application."""

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

    The function updates Streamlit session state only after the complete
    loading and profiling pipeline succeeds.
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
        logger.exception("Unexpected error while preparing uploaded dataset")
        st.error(
            "❌ An unexpected error occurred while preparing the dataset."
        )
        st.exception(exc)
        return

    # Only update state after everything succeeds.
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
        st.metric(
            "Missing Values",
            f"{sum(column.null_count for column in profile.columns):,}",
        )

    with st.expander("Preview data", expanded=False):
        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
        )

    with st.expander("Dataset profile", expanded=False):
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
        )


def _display_chart(chart_spec) -> None:
    """
    Render a chart specification returned by the agent.

    The chart generator intentionally returns a JSON-safe specification
    rather than rendering Streamlit charts itself.
    """

    if not chart_spec:
        return

    chart_type = chart_spec.get("chart_type")
    data = chart_spec.get("data", [])

    if not data:
        return

    st.subheader("📈 Visualization")

    try:
        if chart_type == "bar":
            chart_data = {
                item["category"]: item["value"]
                for item in data
            }

            st.bar_chart(chart_data)

        elif chart_type == "line":
            import pandas as pd

            chart_data = pd.DataFrame(
                {
                    "x": [item["x"] for item in data],
                    "y": [item["y"] for item in data],
                }
            )

            chart_data = chart_data.set_index("x")

            st.line_chart(chart_data)

        elif chart_type == "scatter":
            import pandas as pd

            chart_data = pd.DataFrame(
                {
                    "x": [item["x"] for item in data],
                    "y": [item["y"] for item in data],
                }
            )

            st.scatter_chart(chart_data)

        elif chart_type == "pie":
            # Streamlit does not provide a native pie-chart component.
            # Display the underlying chart data in a readable form.
            import pandas as pd

            chart_data = pd.DataFrame(
                {
                    "Category": [item["label"] for item in data],
                    "Value": [item["value"] for item in data],
                }
            )

            st.dataframe(
                chart_data,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Pie-chart specification generated successfully. "
                "The current Streamlit renderer displays the underlying "
                "category/value data."
            )

        else:
            logger.warning(
                "Unknown chart type returned by chart generator: %s",
                chart_type,
            )

    except Exception as exc:
        logger.exception("Failed to render chart")
        st.warning(
            f"⚠️ The analysis succeeded, but the chart could not be "
            f"rendered: {exc}"
        )


def _run_analysis(question: str, chart_type: str | None) -> None:
    """Run the complete DataAnalystAgent pipeline."""

    agent = st.session_state.agent

    if agent is None:
        st.error("Please upload a CSV file before asking a question.")
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
            logger.exception("Explanation generation failed")
            st.error(
                "⚠️ The SQL analysis completed, but generating the "
                "explanation failed."
            )
            st.exception(exc)
            return

        except Exception as exc:
            logger.exception("Unexpected agent execution error")
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

    st.subheader("🔎 Generated SQL")

    st.code(
        result.sql,
        language="sql",
    )

    st.subheader("📊 Query Result")

    query_result = result.query_result

    if query_result is not None:

        st.caption(
            f"Rows returned: {query_result.row_count:,} | "
            f"Truncated: {query_result.truncated}"
        )

        st.dataframe(
            query_result.dataframe,
            use_container_width=True,
        )

    if result.explanation:
        st.subheader("💡 Analysis")

        st.write(result.explanation)

    if result.chart:
        _display_chart(result.chart)


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

            if (
                current_filename
                != st.session_state.uploaded_filename
            ):
                _load_uploaded_file(uploaded_file)

        st.divider()

        st.header("📈 Chart")

        chart_option = st.selectbox(
            "Chart type",
            options=[
                "Automatic",
                "Bar",
                "Line",
                "Pie",
                "Scatter",
            ],
            index=0,
        )

        chart_type_map = {
            "Automatic": None,
            "Bar": "bar",
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
    # Dataset state
    # ----------------------------------------------------------------------

    if st.session_state.dataframe is None:

        st.info(
            "👈 Upload a CSV file from the sidebar to begin."
        )

        st.markdown(
            """
### What this app can do

1. 📂 Load your CSV dataset
2. 🔍 Profile its columns and data
3. 🤖 Convert your question into SQL using Amazon Bedrock
4. 🛡️ Validate the generated SQL through the security layer
5. 🦆 Execute safe SQL against DuckDB
6. 📊 Generate a visualization
7. 💡 Explain the result in plain English
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

    st.subheader("💬 Ask a question about your data")

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What are the top 10 products by total sales?"
        ),
        height=100,
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


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    main()