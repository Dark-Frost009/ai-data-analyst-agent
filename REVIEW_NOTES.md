# Migration and review verification — September 7, 2026

Changes are saved in the existing checkout. No push, deployment, account
closure or live model request was performed. The pre-existing monetary
no-match fix was preserved in intent and strengthened, not claimed as new.

| Review finding | Verified outcome |
| --- | --- |
| Monetary regex could affect unrelated numeric filters | Replaced expression-span regex with conservative SQL AST matching. No match, unknown vocabulary, multiple quantities/predicates, ratios and counts leave SQL unchanged. Regression tests protect Year/Shows and projection/literal cases. This remains a narrow heuristic, not semantic proof. |
| UI lacked behavioral coverage | Added AppTest upload-boundary fixtures exercising real CSV parsing, profiling, agent/SQL execution, result/chart display, reset/replacement and all seven analysis errors. Remote text is stubbed. File attachment itself is not simulated by AppTest. |
| Mocked tests do not establish model accuracy | Confirmed. Synthetic CSV and handwritten response/expected-row fixtures exercise the actual planner/validator/executor. Live accuracy remains unmeasured. |
| Security walk did nothing | Removed the no-op walk; existing conversion/function tests continue to pass. |
| Empty prompt/UI packages | Extracted planning system instructions, CSS and analysis action. Main still owns the remaining presentation; no wholesale rewrite. |
| Raw keyword checks rejected strings | Planner uses SQL parsing for statement shape; independent AST security checks remain. DELETE and semicolons inside literals are tested. |
| CTE scope imprecision | Table authorization now uses SQLGlot lexical scopes. Nested CTEs cannot authorize outer physical tables; CTE shadowing tests pass. |
| Set operations undocumented | Documented and explicitly rejected, including nested UNION/INTERSECT/EXCEPT. |
| Production MagicMock fallback | Removed. Tests now configure the context-managed executor's table name. |
| Duplicate normalized headers | Reject case/whitespace collisions and blank normalized headers with a parsing error. |

Additional verified issue: the function allowlist rejected ordinary AND/OR
predicates because SQLGlot models them as function nodes. These connectors
now pass while child functions and tables remain independently checked.
Semantic execution fixtures exposed this issue. Failed analyses now also
clear old results instead of displaying a stale answer for the new question.

Groq is behind an LLMClient Protocol and explicit factory. The application,
requirements, configuration and deployment instructions no longer depend
on the former cloud provider. SDK 1.7.0 is pinned. The configurable default
is openai/gpt-oss-20b, checked against official model/free-limit listings.
No automatic retries, fallback providers, tools or model substitution are
used. Missing keys do not prevent startup/uploads. Responses must be complete;
provider error bodies and keys are omitted from application errors.

Production fails closed if APP_ACCESS_PASSWORD is blank. README documents
Community Cloud Python 3.11, private top-level Secrets, limited free quotas,
model availability, the synthetic live-check workflow and privacy boundaries.
The profile includes first-five-row values and statistics; explanation sends
up to 20 result rows. This is not a local-only or automatically redacted app.

Validation completed locally:

- Full suite: 357 passed, 88.43% total coverage on Python 3.12.7.
- Main UI coverage: 76%; extracted analysis action: 96%.
- Real Streamlit process with no Groq key: health endpoint returned HTTP 200 / ok; test process then stopped.
- Python compilation and dependency consistency checks passed.
- Git diff whitespace check passed; no old provider SDK/config identifiers remain in application, tests or deployment files.

Commands: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`,
`python -m compileall -q app`, and `python -m pip check` using the project
virtual environment. Cache provider was disabled because the existing
.pytest_cache directory is inaccessible in this workspace.

Not validated: live Groq authentication/model output/latency/account quota,
Community Cloud deployment, Docker image build, or Python 3.11 execution.
Docker and Python 3.11 were not found in the available commands/common local
installation locations. Existing CI still targets Python 3.11 and Docker;
it was not triggered because no push was authorized.
