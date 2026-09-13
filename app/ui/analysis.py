"""Analysis action and safe user-facing error mapping."""
import streamlit as st
from app.core.agent import (AgentCapacityError, AgentExecutionError,
    AgentPlanningError, AgentValidationError)
from app.core.query_planner import MAX_CONVERSATION_TURNS
from app.utils.logger import get_logger
logger = get_logger(__name__)

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

    st.session_state.last_result = None

    with st.spinner("🤖 Analyzing your question..."):

        try:

            result = agent.run(
                question=question,
                generate_explanation=True,
                generate_chart_spec=True,
                chart_type=chart_type,
                conversation_context=(
                    st.session_state.conversation_history
                ),
            )

        except AgentCapacityError:

            logger.warning("Analysis request rejected because capacity is full")

            st.warning(
                "⏳ The app is busy processing another analysis. "
                "Please try again in a moment."
            )
            return

        except AgentPlanningError as exc:

            logger.exception("Agent planning failed")

            st.error(
                "❌ The AI model could not generate a valid analysis plan."
            )
            return

        except AgentValidationError as exc:

            logger.exception("SQL validation failed")

            st.error(
                "🛡️ The generated SQL did not pass the security checks."
            )
            return

        except AgentExecutionError as exc:

            logger.exception("SQL execution failed")

            st.error(
                "❌ The validated SQL could not be executed."
            )
            return

        except Exception as exc:

            logger.exception(
                "Unexpected agent execution error"
            )

            st.error(
                "❌ The analysis could not be completed."
            )
            return

    st.session_state.last_result = result

    # Only completed, validated work becomes context for a later follow-up.
    # The planner itself applies the same bound again as a defense in depth.
    if result.validation.is_valid and result.validation.cleaned_sql:
        history = list(st.session_state.conversation_history)
        history.append(
            {
                "question": result.question,
                "sql": result.validation.cleaned_sql,
            }
        )
        st.session_state.conversation_history = history[
            -MAX_CONVERSATION_TURNS:
        ]

        logger.info(
            "Conversation context updated | retained_turns=%d",
            len(st.session_state.conversation_history),
        )
