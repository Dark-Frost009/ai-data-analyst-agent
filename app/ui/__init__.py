"""
Presentation layer.

Contains only Streamlit rendering code (widgets, layout, chat display).
No business logic, data processing, or LLM calls belong here — those live
in `app.core`. UI modules call into `app.core` and render the results.
"""
