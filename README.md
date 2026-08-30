# 🤖 AI Data Analyst Agent

An AI-powered data analysis application that allows users to upload CSV datasets and ask analytical questions using natural language.

The application converts natural-language questions into SQL, validates the generated SQL through a security layer, executes the query using DuckDB, and presents the results with explanations and visualizations.

---

## 🚀 Features

### 📊 Natural Language Data Analysis

Ask questions about your dataset in plain English instead of writing SQL manually.

Examples:

- What is the average percentage increase?
- What are the top 5 movies by gross?
- Which year had the highest average revenue?
- Show the total gross by genre.
- What are the lowest 5 values?

---

### 🧠 AI-Powered SQL Generation

The application uses **Amazon Bedrock** to convert natural-language questions into DuckDB-compatible SQL queries.

The Query Planner:

- Understands the user's analytical question.
- Inspects the dataset profile.
- Uses only available columns.
- Preserves exact column names.
- Generates read-only SQL.
- Handles numeric and text-based values intelligently.

---

### 🛡️ SQL Security Validation

Generated SQL is treated as untrusted input and passed through a dedicated security validation layer before execution.

The security layer uses **SQLGlot** to parse the generated SQL into an Abstract Syntax Tree (AST).

It validates:

- SQL statement type
- Table references
- Function usage
- Multiple statements
- Dangerous SQL operations
- Query limits
- Type conversions
- Read-only execution

Unsafe SQL is rejected before it reaches DuckDB.

---

### 🔢 Robust Data Handling

The Query Planner is designed to handle real-world CSV data where values may not be stored in clean numeric formats.

Supported cases include:

- Numeric VARCHAR columns
- Numbers containing commas
- Currency values
- Percentage values
- Negative financial values using parentheses
- NULL values
- Date columns stored as VARCHAR
- Timestamp columns stored as VARCHAR

Examples of supported values:

```text
1,234,567.89
$1,234.50
₹1,25,000
25%
(1,234.50)
```

Safe conversions use DuckDB-compatible functions such as:

```sql
TRY_CAST(...)
REPLACE(...)
CASE
```

The system deliberately avoids unsafe or unsupported functions such as `REGEXP_REPLACE`.

---

## 🏗️ Architecture

The application follows a controlled pipeline that separates AI reasoning, SQL validation, execution, and result presentation.

```text
                    User
                      │
                      ▼
            ┌──────────────────┐
            │    Streamlit UI  │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  Data Loader     │
            │  & Profiler      │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │   AI Agent       │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  Query Planner   │
            │ Amazon Bedrock   │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ SQL Security     │
            │    Validator     │
            │    SQLGlot       │
            └────────┬─────────┘
                     │
                Valid SQL
                     │
                     ▼
            ┌──────────────────┐
            │  SQL Executor    │
            │     DuckDB       │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Query Results    │
            └───────┬──────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 ┌─────────────────┐   ┌─────────────────┐
 │ Chart Generator │   │    Explainer    │
 └────────┬────────┘   └────────┬────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
              Results in UI
```

---

## 🔄 Request Flow

1. The user uploads a CSV dataset.
2. The Data Loader reads and prepares the dataset.
3. The Data Profiler analyzes the dataset structure and available columns.
4. The user asks an analytical question in natural language.
5. The Query Planner sends the question and dataset profile to Amazon Bedrock.
6. Amazon Bedrock generates a DuckDB-compatible SQL query.
7. The generated SQL is passed to the SQL Security Validator.
8. SQLGlot parses the SQL into an Abstract Syntax Tree (AST).
9. The security layer validates the SQL against the project's safety rules.
10. Unsafe or invalid SQL is rejected.
11. Validated SQL is passed to the SQL Executor.
12. DuckDB executes the read-only analytical query.
13. The result is processed for visualization and explanation.
14. The Streamlit interface displays the analytical result.

---

## 🔐 Security Design

Security is a core part of the application architecture.

The AI-generated SQL is **never trusted automatically**.

The application follows this flow:

```text
Natural Language
       │
       ▼
 Amazon Bedrock
       │
       ▼
 Generated SQL
       │
       ▼
 SQLGlot Parser
       │
       ▼
 Security Validation
       │
   ┌───┴───┐
   │       │
Unsafe    Safe
   │       │
   ▼       ▼
Reject   DuckDB
           │
           ▼
         Result
```

### Security Rules

The validator:

- Allows only read-only `SELECT` / `WITH` queries.
- Rejects multiple SQL statements.
- Restricts table references to approved tables.
- Restricts SQL functions to an explicit allowlist.
- Blocks data modification operations.
- Blocks database definition operations.
- Blocks file-access operations.
- Blocks extension installation/loading.
- Blocks external database attachment.
- Enforces a maximum query result limit.
- Validates SQL before execution.

Examples of blocked operations include:

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
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── __init__.py
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
│   ├── test_chart_generator.py
│   ├── test_data_loader.py
│   ├── test_data_profiler.py
│   ├── test_explainer.py
│   ├── test_llm_client.py
│   ├── test_query_planner.py
│   ├── test_schemas.py
│   ├── test_security.py
│   └── test_sql_executor.py
│
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application language |
| Streamlit | Web interface |
| Amazon Bedrock | AI-powered SQL generation |
| Amazon Nova | Large Language Model |
| DuckDB | Local analytical SQL execution |
| SQLGlot | SQL parsing and security validation |
| Pandas | Data processing |
| Pydantic | Data validation and schemas |
| Pytest | Automated testing |

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

Create a `.env` file based on `.env.example`.

Example:

```text
AWS_REGION=ap-south-1
BEDROCK_MODEL_ID=apac.amazon.nova-lite-v1:0
```

Configure your AWS credentials using the AWS CLI or your preferred secure AWS credential mechanism.

**Never commit `.env` or AWS credentials to GitHub.**

---

## ▶️ Running the Application

Start Streamlit with:

```powershell
streamlit run app/main.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## 🧪 Running Tests

Run the complete test suite with:

```powershell
pytest -v
```

Run a specific test module:

```powershell
pytest tests/test_security.py -v
```

Run Query Planner tests:

```powershell
pytest tests/test_query_planner.py -v
```

---

## 💬 Example Questions

Once a dataset is uploaded, users can ask questions such as:

```text
What is the average percentage increase?
```

```text
What are the top 5 movies by actual gross?
```

```text
Which year had the highest average gross?
```

```text
What is the total gross by genre?
```

```text
Show me the lowest 5 values.
```

```text
What is the average adjusted gross?
```

The application generates SQL dynamically based on the uploaded dataset's schema.

---

## 📈 Example Analytical Flow

For a question such as:

```text
What is the average percentage increase?
```

The system can generate a query that safely:

1. Reads the relevant columns.
2. Removes currency symbols when necessary.
3. Removes thousands separators.
4. Converts text values into numeric values using `TRY_CAST`.
5. Calculates the percentage increase.
6. Handles zero denominators safely.
7. Calculates the average.
8. Returns the result to the application.

---

## 🧠 Design Principles

The project follows several important engineering principles.

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

### Defense in Depth

The application does not rely only on the LLM prompt for security.

Security is enforced independently by the SQL validation layer.

```text
LLM Instructions
      +
Planner Validation
      +
SQLGlot AST Validation
      +
Function Allowlist
      +
Table Allowlist
      +
LIMIT Enforcement
      =
Safer SQL Execution
```

### Fail Safely

Invalid or unsafe queries are rejected instead of being executed.

### Schema Awareness

The AI receives the dataset profile and is instructed to use only real columns from the uploaded dataset.

### Read-Only Analytics

The application is designed for analytical queries and does not modify the underlying dataset.

---

## 🧪 Testing Strategy

The project includes tests covering the major application components.

Test coverage includes:

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

The security tests specifically verify that unsafe SQL is rejected.

---

## 🔒 Data & Credential Safety

Uploaded datasets are intentionally excluded from Git tracking through `.gitignore`.

The repository also ignores:

```text
.env
.venv/
__pycache__/
.pytest_cache/
data/*.csv
data/*.parquet
.streamlit/secrets.toml
```

AWS credentials and other secrets should never be stored directly in the repository.

---

## 🚧 Current Limitations

The current version is primarily designed for structured CSV-based analytical workflows.

Potential future improvements include:

- Support for larger datasets
- More advanced visualizations
- Conversational follow-up questions
- Query result caching
- More sophisticated data type inference
- Improved error recovery
- Additional SQL dialect support
- Persistent analytical sessions
- Deployment to AWS
- Authentication and user management

---

## 🗺️ Future Roadmap

```text
Current
  │
  ├── CSV Upload
  ├── Dataset Profiling
  ├── Natural Language Questions
  ├── AI SQL Generation
  ├── SQL Security Validation
  ├── DuckDB Execution
  ├── Visualization
  └── AI Explanation
        │
        ▼
Future
  │
  ├── Conversational Analytics
  ├── Multiple Dataset Support
  ├── Advanced Visualization
  ├── Query History
  ├── Result Caching
  ├── Cloud Deployment
  ├── Authentication
  └── Production Monitoring
```

---

## 📌 Why This Project?

This project demonstrates how an AI-powered analytical application can combine:

- Generative AI
- Natural Language Processing
- SQL generation
- Data engineering
- Data analysis
- Software engineering
- Application security
- Automated testing
- Cloud AI services

Rather than allowing an LLM to directly execute generated SQL, the application introduces a dedicated validation layer between AI generation and database execution.

This makes the architecture more suitable for building safer AI-powered data applications.

---

## 👨‍💻 Author

**Sayantan Chaklader**

AI/ML • Data Analytics • Python

GitHub:  
https://github.com/Dark-Frost009

---

## ⭐ Project Status

**Status: Working Prototype**

The application has been tested with multiple natural-language analytical questions and successfully generates and executes analytical SQL queries against uploaded datasets.

The current version is being actively developed and improved.

---

## 📄 License

This project is currently intended for educational, portfolio, and demonstration purposes.