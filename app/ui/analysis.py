"""Analysis action and safe user-facing error mapping."""
import streamlit as st
from app.core.agent import (AgentCapacityError, AgentExecutionError,
    AgentPlanningError, AgentValidationError, AgentDataError)
from app.core.query_planner import MAX_CONVERSATION_TURNS
from app.core.llm_client import LLMCredentialsError, LLMAccessDeniedError, LLMThrottlingError, LLMAPIError
from app.utils.logger import get_logger
logger = get_logger(__name__)

def _planning_error_message(error):
    messages = {
        LLMCredentialsError: "The AI service credentials are missing or invalid. The app owner needs to check the private deployment settings.",
        LLMAccessDeniedError: "The configured AI model is unavailable for this account. The app owner needs to check model access.",
        LLMThrottlingError: "The AI request quota or hourly service budget has been reached. Try again later.",
        LLMAPIError: "The AI service could not be reached or rejected the request. Try again shortly; if it persists, contact the app owner.",
    }
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for kind, message in messages.items():
            if isinstance(error, kind):
                return message
        error = error.__cause__ or error.__context__
    return "❌ The AI model could not generate a valid analysis plan."


def _run_analysis(
    question: str,
    chart_type: str | None,
) -> None:
    """Run the complete DataAnalystAgent pipeline."""

    if st.session_state.get("upload_error", False):
        st.error("The selected CSV was not loaded. Choose a valid file before analyzing.")
        return
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

        except AgentDataError as exc:
            logger.warning("Analysis stopped by conversion fidelity check")
            st.warning(str(exc))
            return

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
                _planning_error_message(exc)
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
