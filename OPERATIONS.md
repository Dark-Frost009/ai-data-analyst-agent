# Operating the controlled production demo

This deployment supports a small, access-controlled audience. It has no
availability SLA, durable user accounts, permanent dataset storage or shared
multi-process quota service. Do not advertise unlimited uploads or enterprise
readiness. The default 100 MB ceiling is subject to actual host resources.

## Release and configuration

The hosted app tracks `main`, with entry point `app/main.py`. Run the full test
suite before pushing and require both GitHub jobs (tests and Docker health) to
pass. Streamlit may deploy a push before CI finishes; revert the release if a
required job fails. For stricter release gating, move the app to a protected
release branch after an owner configures branch protection.

Keep `APP_ENV=production`, `APP_ACCESS_PASSWORD`, and `GROQ_API_KEY` in the
hosting secret settings. Never paste credentials into tickets or logs. Docker
defaults to production and fails closed without an access password. The shared
demo code is a lightweight access gate, not per-user identity or an audit trail.
Do not use this demo to process sensitive production data without a separate
privacy and access review appropriate to your organization.

Current defaults are:

| Control | Default |
|---|---:|
| Upload ceiling (app and Streamlit config) | 100 MB |
| Large-dataset rows / columns | 2,000,000 / 200 |
| Temporary dataset storage / process storage | 512 MB / 1,024 MB |
| Reserved DataFrame memory / process memory | 128 MB / 256 MB |
| DuckDB query memory / threads | 512 MB / 2 |
| Simultaneous analyses / preparations | 1 / 1 |
| Preparation timeout / query timeout | 180 s / 30 s |
| Provider connect timeout / read timeout | 5 s / 45 s |
| Rolling AI request cap per process | 60 per hour |
| Large-file displayed result size / row limit | 16 MB / 10,000 |

An analysis normally makes two AI calls, one for planning and one for its
explanation. The hourly cap counts attempted provider requests, including
failures. It resets on process restart; multiple workers have separate budgets.
Groq's own limits may be stricter and may be shared with other applications
using the account. Check actual account usage in the provider console before
sharing the demo. Do not raise limits to bypass a provider quota.

## Release smoke test

Unlock the hosted app directly in the browser. Upload a synthetic CSV and
verify the displayed environment is production, the uploader shows 100 MB,
and large-file profiles disclose their sample size. For the 12,000-row fixture
used during this release, employees joining after 2022 must return North:
4,000 / salary 400,000 and South: 4,000 / salary 800,000. Check the result table,
explanation, chart, and direct typed-date predicate in Generated SQL.

Use two sessions with distinct synthetic datasets to verify results remain
isolated. Competing analyses should return a busy response, then succeed when
retried after the first finishes. Do not load-test a free shared host with
unbounded concurrent uploads: Streamlit keeps incoming uploads in memory
before the application's limits can run.

## Monitoring

The Deployment availability workflow checks `/_stcore/health` every 30 minutes
and can be run manually. It uses no credentials or AI requests. GitHub scheduled
jobs can be delayed or disabled; review the Actions tab and configure your
existing GitHub notification preferences for failed runs. This is an
availability probe, not an SLA or a test of Groq, authentication, or SQL answers.

Logs include random analysis IDs, request outcomes, timings, provider HTTP
status and token usage when available. They omit SQL, user questions, provider
response bodies and tracebacks. Dataset filenames and column-normalization
metadata may still appear. Restrict log access and follow your retention policy.
No external log service or paid monitoring account is required.

## Recovery

| Symptom | Action |
|---|---|
| Busy response | Wait for the running analysis/preparation, then retry. |
| AI quota / hourly cap | Stop retrying; wait for the window or inspect account limits. |
| Credentials/model access warning | Owner checks private credentials and model access. |
| Provider unreachable | Check Groq status/network; retry later without changing security checks. |
| Oversized file/query | Use fewer rows/columns or aggregate the result; do not blindly raise limits. |
| Storage full | Clear unused datasets; a process restart can recover abandoned sessions. |
| Bad replacement CSV | Retry, choose a valid CSV, or remove the selection to resume the old dataset. |
| Host resource-limit error | Lower the workload or limits and restart; investigate memory before expanding. |
| New release breaks checks | Revert its commit and push the revert; keep credentials unchanged. |

On GitHub, identify the failed release SHA. Use `git revert <release-sha>` and
push the resulting revert after reviewing it; never force-push or reset away
unrelated work. Verify CI, the hosted health endpoint and the synthetic query
again. Keep the previous known-good revision in the release record.

Each runtime holds an OS lock on its private temporary directory. Dataset
clear/replacement and garbage collection remove session files; a process
startup/first disk upload reaps application-marked directories whose owner lock
is no longer held. Active processes are skipped. Legacy unmarked temporary
folders are not deleted automatically. Query timeouts retain data until their
worker has stopped. Uploaded data are temporary, so users must keep originals.

Dependabot already checks Python, GitHub Actions and Docker updates weekly.
CI checks installed application/test dependencies against OSV and fails on
known advisories (pip/setuptools build tooling excluded). A failed or unavailable
advisory service also fails the check; investigate before releasing.
Re-run tests and advisory checks when accepting dependency changes. An advisory
check covers known issues in the versions checked, not a penetration test.

Sources: [Streamlit resource guidance](https://docs.streamlit.io/knowledge-base/deploy/resource-limits),
[deployment secrets](https://docs.streamlit.io/deploy/concepts/secrets), and
[Groq rate limits](https://console.groq.com/docs/rate-limits).


## Release verification (October 2, 2026)

The patched dependency environment passed **468 tests**, with **89.12% coverage**
in the actual repository. The OSV check reported no known advisories in 59
installed application/test packages (pip/setuptools excluded), and dependency
resolution passed `pip check`. Patched direct pins are Streamlit 1.54.0,
python-dotenv 1.2.2 and pytest 9.0.3.

A local bounded two-session check used two 25.18 MB CSVs, each with 100,000
rows. The sessions returned independent totals of 200,000 and 300,000, returned
a busy response for an overlapping analysis, recovered afterward, and removed
all temporary data/reservations. The check took about 5.33 seconds locally.
The earlier local 96.7 MB upload and live Groq browser test also passed. These
measurements do not guarantee equivalent performance on free hosting.

Hosted results and the release commit/CI links are recorded separately after
the push and deployed smoke test.
