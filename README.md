# 🤖 AI Data Analyst Agent

[![Test](https://github.com/Dark-Frost009/ai-data-analyst-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Dark-Frost009/ai-data-analyst-agent/actions/workflows/ci.yml)

[**Open the live demo**](https://groq-data-analyst.streamlit.app/) — access code required.

An AI-powered data analysis application that allows users to upload CSV datasets and ask analytical questions using natural language.

The application converts natural-language questions into DuckDB-compatible SQL using Groq, validates the generated SQL through an independent security layer, executes the validated query with DuckDB, and presents the results with tables, visualizations, and AI-generated explanations.

---

## 🚀 Features

### 📊 Natural-Language Data Analysis

Ask questions about an uploaded dataset using plain English instead of writing SQL manually.

Example questions:

```text
How many records are there?
```

```text
What are the top 3 artists by average gross?
```

```text
Which tours had an average gross greater than $5 million?
```

```text
Which year had the highest total gross?
```

```text
Show the average gross by artist.
```

The application dynamically generates SQL based on the uploaded dataset's schema.

### 💬 Session-Local Follow-Up Questions

After a successful analysis, ask a follow-up such as "Now show only the
top 2." The planner receives up to three prior successful question and
validated-SQL pairs from the current browser session.

Conversation context is bounded, can be cleared from the sidebar, and is
automatically reset when the active dataset changes. It is not durable
storage, and every new SQL statement still passes the full validation and
execution pipeline.

---

### 🧠 AI-Powered SQL Generation

The application uses **Groq** to translate natural-language analytical questions into DuckDB-compatible SQL.

The Query Planner:

- Understands analytical intent.
- Inspects the dataset profile.
- Uses only columns that exist in the dataset.
- Preserves exact column names.
- Generates read-only SQL.
- Handles aggregation, filtering, grouping, ordering, ranking, and limits.
- Handles numeric values stored as text.
- Handles currency-formatted values.
- Handles date and timestamp values stored as text.
- Applies deterministic corrections for explicit monetary thresholds.

The LLM is responsible for generating the analytical query, while security validation is handled independently.

---

### 🛡️ SQL Security Validation

Generated SQL is treated as **untrusted input**.

Before execution, every generated query passes through a dedicated SQL security validation layer.

The security layer uses **SQLGlot** to parse SQL into an Abstract Syntax Tree (AST) and validates the query against the application's safety rules.

The validator checks:

- SQL statement type
- Table references
- Function usage
- Multiple SQL statements
- Dangerous SQL operations
- Query result limits
- Read-only execution
- Type conversions
- Allowed SQL functions
- Unqualified in-memory table references only

Unsafe SQL is rejected before reaching DuckDB.

This creates a security boundary between the LLM and the database.

DuckDB is also configured as a second line of defense: external
file/network access and temporary-disk spilling are disabled, while memory,
thread, result-size, and query-time limits constrain execution resources.

---

### 🧮 Robust Numeric and Data Handling

Real-world CSV datasets frequently contain numbers stored as strings rather than clean numeric values.

The Query Planner supports patterns such as:

```text
1,234,567.89
$1,234.50
25%
```

For example, numeric text can be safely converted using:

```sql
TRY_CAST(...)
```

Thousands separators can be removed using:

```sql
REPLACE(...)
```

Currency values can be cleaned before conversion:

```sql
TRY_CAST(
    REPLACE(
        REPLACE(
            CAST("Average gross" AS VARCHAR),
            ',',
            ''
        ),
        '$',
        ''
    ) AS DOUBLE
)
```

The application deliberately avoids unsupported or unsafe SQL functions such as `REGEXP_REPLACE`.

---

### 💰 Explicit Monetary Threshold Handling

The Query Planner includes deterministic handling for explicit monetary quantities.

Examples:

```text
$5 million
$10 million
$2.5 million
$750 thousand
$1 billion
$5M
$750K
```

These are converted to their mathematical equivalents:

```text
$5 million      → 5000000
$10 million     → 10000000
$2.5 million    → 2500000
$750 thousand   → 750000
$1 billion      → 1000000000
```

This is a narrow safeguard, not proof of semantic correctness. It only changes a single unambiguous monetary comparison. Unknown terms (including box office, budget and turnover), multiple quantities, and ambiguous expressions leave SQL unchanged. Always inspect important results.

---

## 🏗️ Architecture

The application follows a controlled pipeline that separates data processing, AI reasoning, security validation, SQL execution, and result presentation.

```text
                         User
                           │
                           ▼
                  ┌─────────────────┐
                  │  Streamlit UI   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Data Loader   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Data Profiler  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    AI Agent     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Query Planner  │
                  │      Groq       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ SQL Security    │
                  │ Validator       │
                  │    SQLGlot      │
                  └────────┬────────┘
                           │
                     Valid SQL
                           │
                           ▼
                  ┌─────────────────┐
                  │  SQL Executor   │
                  │     DuckDB      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Query Results   │
                  └───────┬─────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
        ┌─────────────────┐ ┌─────────────────┐
        │ Chart Generator │ │    Explainer    │
        └────────┬────────┘ └────────┬────────┘
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    Results in UI
```

---

## 🔄 Request Flow

1. The user uploads a CSV dataset.
2. The Data Loader reads and prepares the dataset.
3. The Data Profiler analyzes the dataset structure and columns.
4. The user asks an analytical question in natural language. For a
   follow-up, the planner may also receive bounded context from prior
   successful turns in the same browser session.
5. The Query Planner receives the question and dataset profile.
6. Groq generates a DuckDB-compatible SQL query.
7. The generated SQL is treated as untrusted input.
8. SQLGlot parses and validates the SQL.
9. Security rules verify that the query is read-only and accesses only approved resources.
10. Unsafe or invalid SQL is rejected.
11. Valid SQL is passed to the SQL Executor.
12. DuckDB executes the analytical query.
13. The result is processed by the application.
14. The Chart Generator determines whether a visualization is useful.
15. The Explainer generates a concise analytical interpretation.
16. The Streamlit interface presents the result.

---

## 🔐 Security Model

The application follows a **defense-in-depth** approach.

The LLM prompt provides instructions, but the application does not rely on the LLM to enforce security.

```text
LLM Instructions
       │
       ▼
Query Planner
       │
       ▼
SQLGlot AST Validation
       │
       ├── Statement validation
       ├── Table validation
       ├── Function allowlist
       ├── Dangerous operation checks
       ├── Multiple statement checks
       └── Result limit enforcement
       │
       ▼
    Safe SQL
       │
       ▼
    DuckDB
```

The validator rejects operations such as:

```text
INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
TRUNCATE
COPY
ATTACH
DETACH
INSTALL
LOAD
CALL
EXPORT
IMPORT
```

The application is designed specifically for read-only analytical workloads.

---

## 🧪 Example Analytical Queries

The agent has been tested with analytical questions involving different SQL patterns.

### Record counting

Question:

```text
How many records are there?
```

Generated SQL:

```sql
SELECT COUNT(*) FROM "dataset"
```

---

### Ranking with aggregation

Question:

```text
What are the top 3 artists by average gross?
```

Generated SQL:

```sql
SELECT
    "Artist",
    AVG(
        TRY_CAST(
            REPLACE(
                REPLACE(
                    CAST("Average gross" AS VARCHAR),
                    ',',
                    ''
                ),
                '$',
                ''
            ) AS DOUBLE
        )
    ) AS "Average Gross"
FROM "dataset"
GROUP BY "Artist"
ORDER BY "Average Gross" DESC
LIMIT 3
```

This demonstrates:

- `AVG`
- `GROUP BY`
- numeric conversion
- text cleaning
- `ORDER BY`
- descending ranking
- `LIMIT`

---

### Monetary filtering

Question:

```text
Which tours had an average gross greater than $5 million?
```

Generated SQL can safely convert the text-based monetary column before applying the threshold:

```sql
SELECT
    "Tour title",
    "Average gross"
FROM "dataset"
WHERE TRY_CAST(
    REPLACE(
        REPLACE(
            CAST("Average gross" AS VARCHAR),
            ',',
            ''
        ),
        '$',
        ''
    ) AS DOUBLE
) > 5000000
```

---

### Grouping and counting

Question:

```text
How many tours are there for each year?
```

Generated SQL:

```sql
SELECT
    "Year(s)",
    COUNT(*) AS tour_count
FROM "dataset"
GROUP BY "Year(s)"
ORDER BY "Year(s)"
```

---

## 📈 Result Presentation

Depending on the analytical result, the application can present:

- Query results in a table
- Generated SQL
- Result metadata
- Visualizations when appropriate
- AI-generated analytical explanations
- Important values and patterns
- Rankings and comparisons
- Limitations of the result

The goal is to provide both the **answer** and enough information to understand how the answer was produced.

---

## 🧩 Project Structure

```text
ai-data-analyst-agent/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── chart_generator.py
│   │   ├── data_loader.py
│   │   ├── data_profiler.py
│   │   ├── explainer.py
│   │   ├── llm_client.py
│   │   ├── query_planner.py
│   │   └── sql_executor.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── query_planner.py
│   │
│   ├── ui/
│   │   ├── analysis.py
│   │   └── styles.css
│   │
│   └── utils/
│       ├── __init__.py
│       ├── analysis_capacity.py
│       ├── logger.py
│       └── security.py
│
├── data/
│   └── sample/
│       └── .gitkeep
│
├── tests/
│   ├── .gitkeep
│   ├── test_agent.py
│   ├── test_analysis_capacity.py
│   ├── test_config.py
│   ├── test_chart_generator.py
│   ├── test_data_loader.py
│   ├── test_data_profiler.py
│   ├── test_explainer.py
│   ├── test_llm_client.py
│   ├── test_query_planner.py
│   ├── test_schemas.py
│   ├── test_security.py
│   ├── test_sql_executor.py
│   └── test_ui_error_handling.py
│
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
├── .dockerignore
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core application language |
| **Streamlit** | Web application interface |
| **Groq** | LLM inference and SQL generation |
| **GPT-OSS 20B (configurable)** | Initial Groq model |
| **DuckDB** | Local analytical SQL execution |
| **SQLGlot** | SQL parsing and security validation |
| **Pandas** | Dataset processing |
| **Pydantic** | Data validation and schemas |
| **Pytest** | Automated testing |

---

## Sharing the source for review

Share this GitHub repository or download its source from GitHub's **Code →
Download ZIP** menu. That download contains the committed source rather than
the contents of your local working folder. No archive is generated by the app.

Avoid compressing the entire local project folder: `.gitignore` controls Git
tracking, not what Windows includes in an archive. Local `.env`, `.venv`,
`.streamlit/secrets.toml`, `.coverage`, caches, logs and temporary scripts must
stay out of shared files. Coverage databases can contain local filesystem
paths and become stale. Use a fresh `pytest -q` report to verify coverage;
the reported test count and percentage depend on the exact revision and run.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dark-Frost009/ai-data-analyst-agent.git
cd ai-data-analyst-agent
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

The app uses Groq through the provider-independent `LLMClient` interface. No
AWS account, SDK, credential file, region or role is required.

For local development copy `.env.example` to `.env`, then enter your key
privately in that ignored file. Never paste credentials into chat or commit them.
Startup, CSV uploads and offline tests work without a key; analysis needs one.

```text
LLM_PROVIDER=groq
LLM_MODEL_ID=openai/gpt-oss-20b
GROQ_API_KEY=
APP_ENV=development
APP_ACCESS_PASSWORD=
```

`openai/gpt-oss-20b` was listed as a production model and in the free-plan
limits on September 7, 2026. The default is configurable, and availability
must be checked for your account before a live demo. The published free
limits at that check were 30 requests/minute, 1,000 requests/day, 8,000
tokens/minute and 200,000 tokens/day. Your console limits take precedence.
Sources: [models](https://console.groq.com/docs/models) and
[rate limits](https://console.groq.com/docs/rate-limits).

Free quotas are limited. A successful analysis normally makes two requests
(planning and explanation), with prompts and completion budgets consuming
quota. The adapter has no automatic retries, paid-provider fallback, tools,
or automatic model substitution. Quota exhaustion requires trying later.
For GPT-OSS it requests low reasoning effort and excludes reasoning from the
returned answer; completion budgets are 2,048 for planning and 1,024 for
explanations. Truncated completions fail safely. These choices have offline
contract tests, not a live model-accuracy claim.
[Groq SDK](https://github.com/groq/groq-python),
[reasoning options](https://console.groq.com/docs/reasoning).

Other settings and defaults are in `.env.example`: upload/result size,
query timeout, one concurrent analysis per process, DuckDB memory/thread
limits, and provider connect/read timeouts. Only `groq` is implemented;
other provider names fail explicitly.

## ▶️ Running locally

```powershell
python -m streamlit run app/main.py
```

Open [localhost:8501](http://localhost:8501). Local development can omit the
access password; shared deployments must set it.

## ☁️ Streamlit Community Cloud

Use this repository, branch `main`, entry point `app/main.py`, and Python
3.11. In the application's **Advanced settings → Secrets** (or its settings
for an existing app), enter top-level TOML values privately:

```toml
APP_ENV = "production"
APP_ACCESS_PASSWORD = "replace-with-your-private-demo-password"
GROQ_API_KEY = "replace-with-your-private-groq-key"
LLM_PROVIDER = "groq"
LLM_MODEL_ID = "openai/gpt-oss-20b"
MAX_CONCURRENT_ANALYSES = "1"
```

These are placeholders, not working credentials. Top-level secrets become
environment variables. Never commit `.streamlit/secrets.toml`. Restart the
app after changing settings. In production a missing password locks the
interface; retain this gate for the shared demo. It is a shared access code,
not individual accounts or tenant isolation.
[Streamlit secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management),
[top-level environment variables](https://docs.streamlit.io/develop/concepts/connections/secrets-management).

After configuring privately, use the synthetic evaluation CSV under
`tests/fixtures/` to check real answers against the expected results. Model
access, actual response quality, latency and account quotas need this live
check; passing offline tests cannot establish them.

## 🐳 Running with Docker

```powershell
docker build -t ai-data-analyst-agent .
docker run --rm -p 8501:8501 --env-file .env ai-data-analyst-agent
```

Create your private `.env` first. For a shared container set
`APP_ENV=production` and `APP_ACCESS_PASSWORD`. Do not mount any cloud
credential directories. The image runs as a non-root user, excludes local
secrets/data, and checks `/_stcore/health`. Open
[localhost:8501](http://localhost:8501).

---

## 🧪 Testing

The project includes automated tests covering the major application components.

Run the complete test suite:

```powershell
pytest -q
```

The suite measures coverage for `app/` and enforces a minimum total coverage
of **70%**. The exact result is reported by every test run.

The offline semantic fixtures are in `tests/fixtures/analysis.csv` and
`analysis_cases.json`. Their response SQL is handwritten; expected rows are
checked through the real planner, security validator and DuckDB executor.
They cover counting, aggregate ranking, monetary/year filters, string
literals and empty results. They do not measure Groq's question-to-SQL
accuracy. The UI tests stub the unsupported AppTest upload widget boundary
and remote model, while exercising CSV parsing, analysis buttons, results,
charts, all seven error branches, replacement and reset behavior.

The test suite covers:

- Agent behavior
- Chart generation
- Data loading
- Data profiling
- Explanation generation
- LLM client behavior
- Query planning
- Pydantic schemas
- SQL security validation
- SQL execution
- Configuration validation
- Streamlit-safe error presentation
- Streamlit startup smoke rendering
- Analysis-capacity control
- DuckDB timeout, interruption, and disk-spill protection
- Session-local follow-up context and reset behavior
- Offline end-to-end evaluation scenarios

Security tests specifically verify that unsafe SQL operations are rejected.

Run the application manually after changes that affect the interface:

```powershell
streamlit run app/main.py
```

Useful manual checks include replacing an uploaded CSV with another file of
the same name, clearing the active dataset, an unavailable Groq service,
and a query that reaches the configured result limit.

GitHub Actions runs automatically on every push and pull request. It executes
the test suite with the coverage floor, builds the Docker image, starts the
container, and waits for Docker to report it as healthy.

---

## 🚢 Deployment Checklist

- Keep `APP_ENV=production` and a strong private `APP_ACCESS_PASSWORD`.
- Set `GROQ_API_KEY` privately and verify the model in your account.
- Run `pytest -q` on Python 3.11 and review the live synthetic-data answers.
- Check resource limits and quota before sharing the demo.
- For Docker hosting, build and verify the container health endpoint.
- Use only data you are authorized to send to the hosting service and Groq.

No deployment or account changes are performed by the offline test suite.

---

## 🧱 Design Principles

### Separation of Responsibilities

Each component has a focused responsibility:

```text
Data Loader
     ↓
Data Profiler
     ↓
Query Planner
     ↓
Security Validator
     ↓
SQL Executor
     ↓
Result Processing
     ↓
Visualization + Explanation
```

---

### Defense in Depth

Security is implemented independently of the LLM.

```text
LLM Prompt
    +
Planner Rules
    +
SQLGlot AST Validation
    +
Function Allowlist
    +
Table Allowlist
    +
Limit Enforcement
    =
Safer AI-Assisted SQL Execution
```

---

### Fail Safely

Invalid or unsafe SQL is rejected instead of being executed.

---

### Schema Awareness

The AI receives information about the uploaded dataset and is instructed to use only columns that actually exist.

---

### Read-Only Analytics

The application is designed for analytical workloads and does not modify the underlying dataset.

---

## 🔒 Data & Credential Safety

The CSV is processed in the app host's memory with pandas and restricted
DuckDB. It is not uploaded as a file to Groq, but **data values do leave the
host when you analyze**:

- Planning sends your question, the full dataset profile (column names,
  types, counts, numeric statistics and the first five sample rows), and
  up to three previous successful questions and SQL statements.
- Explanation sends the validated SQL, result metadata and up to 20 result
  rows. Values can contain personal or confidential information.
- Follow-up context stays in the browser's server session until cleared or
  lost, and is sent again with follow-up planning requests.

Use synthetic or non-sensitive data for the demo. No automatic redaction
or provider data-retention guarantee is implemented. Review the provider's
[Your Data documentation](https://console.groq.com/docs/your-data) for your
account before uploading sensitive material. Server logs may include SQL,
filenames, normalized headers and execution diagnostics; protect log access.
The adapter does not log API keys or raw provider response bodies.

`.env`, `.streamlit/secrets.toml`, environments and local uploaded data are
ignored by Git. The bundled fixture is deliberately synthetic. Never commit
real credentials or uploaded datasets.

---

## ⚠️ Current Limitations

The current version is primarily designed for structured CSV-based analytical workflows.

Current limitations include:

- Primarily focused on CSV datasets
- Query generation depends on LLM quality
- Complex analytical questions may require additional validation or refinement
- Dataset size is limited by the local execution environment
- A shared password gate is available; individual accounts and tenant isolation are not implemented
- UNION, INTERSECT and EXCEPT are intentionally unsupported
- Monetary correction is conservative and English-only: ambiguous predicates, multiple monetary quantities and unknown terms are left unchanged
- Follow-up context is session-only, limited to three successful turns, and
  is not durable query history
- The app is intended for controlled, trusted CSV uploads rather than
  arbitrary public file-processing workloads

---

## 🗺️ Future Roadmap

Potential future improvements include:

```text
Current
   │
   ├── CSV Upload
   ├── Dataset Profiling
   ├── Natural-Language Questions
   ├── AI SQL Generation
   ├── SQL Security Validation
   ├── DuckDB Execution
   ├── Visualization
   └── AI Explanation
          │
          ▼
Future
   │
   ├── Durable Query History
   ├── Multiple Dataset Support
   ├── Advanced Visualizations
   ├── Query History
   ├── Result Caching
   ├── Authentication
   └── Production Monitoring
```

---

## 💡 Why This Project?

This project demonstrates how generative AI can be integrated into a data-analysis workflow without giving the language model direct control over database execution.

The architecture combines:

- Generative AI
- Natural-language processing
- SQL generation
- Data engineering
- Data analysis
- Application security
- Automated testing
- Cloud AI services

The key architectural principle is the separation between **AI-generated SQL** and **SQL execution**.

Instead of allowing the LLM to directly interact with the database:

```text
Natural Language
      ↓
     LLM
      ↓
Generated SQL
      ↓
Security Validation
      ↓
    DuckDB
      ↓
   Results
```

This provides an additional security boundary and makes the system easier to reason about, test, and extend.

---

## 👨‍💻 Author

**Sayantan Chaklader**

AI/ML • Data Analytics • Python

GitHub:  
https://github.com/Dark-Frost009

---

## 📌 Project Status

**Deployed on Streamlit Community Cloud with Groq and a shared password gate.**

- Live app: [groq-data-analyst.streamlit.app](https://groq-data-analyst.streamlit.app/).
- Local validation (September 8, 2026): 360 tests passed on Python 3.12.7, with 88.49% coverage, including WITH-prefixed query planning and execution.
- [GitHub Actions run #14](https://github.com/Dark-Frost009/ai-data-analyst-agent/actions/runs/34142933104) passed Python 3.11 tests and Docker build/startup health checks for commit `4647d8f`.
- A manual live check using the bundled synthetic CSV returned the expected artists **A and C** for “Which artists in year 2015 had gross over $5 million?” The public password gate was also checked independently.

This is a deployment smoke check, not a comprehensive model-accuracy evaluation.
Offline fixtures use fixed model responses and validate the surrounding pipeline.
Run `pytest -q` for current local results. Remaining work includes individual
user accounts, durable query history and production monitoring.
