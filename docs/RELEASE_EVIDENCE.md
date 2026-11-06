# Risk stress lab: verification evidence

Verified November 5, 2026 (America/Denver). This is evidence for a tested feature branch, not approval for a production trading deployment.

Source baseline: `3bb3b7e45fe2adcad88e6e6be331eac0c7cb1980`. Scope: the implementation and tests in the commit containing this report. A sorted SHA-256 manifest of `app/**/*.py`, `mcp_server/**/*.py`, `tests/**/*.py` and `static/index.html` was captured outside the repository; manifest SHA-256: `E8D16C863324C00DE3CCEE9BFF27B5816C553AD9D196FA13FD240F4BDDC7EB2F`.

Environment: Windows, isolated `.venv`, CPython 3.11.0, NumPy 2.4.6, FastAPI 0.162.0, Pydantic 2.14.3, LangGraph 1.2.20, MCP 2.3.0, pytest 9.2.0 and Playwright 1.64.1 with installed Chrome. Tests use sample market data and mocked external model responses. Local networking was enabled because Windows restricted the API test client's event-loop socket otherwise.

| Area | Status | Evidence / limit |
| --- | --- | --- |
| Unit, API, graph, MCP, provider failures | PASS | `python -m pytest tests/ -q --tb=short`: **102 passed**. Includes distribution fixtures, seed/rate preservation, invalid and missing decisions, reduction-only sizing, rejection and duplicate/concurrent decisions. |
| Rendered critical journeys | PASS | `python -m pytest tests/browser_smoke.py -q`: **2 passed**, at 1440x1080 and 390x844. Actual HTML/JS and FastAPI routes with fixture model responses. Keyboard navigation, locally scrolling table, escaped notes, rejected upsize, smaller approval, cancellation; no JavaScript page errors. Screenshots visually inspected. |
| Reduced motion | PASS | Both browser journeys use reduced-motion preference. Existing CSS/Chart.js reduced-motion handling retained. |
| Python compilation and diff hygiene | PASS | `python -m compileall -q app mcp_server`; `git diff --check`. No configured type checker or linter exists in the repository; compilation is not a type/lint claim. |
| Dependency compatibility | PASS | `uv pip check --python .venv/Scripts/python.exe`: 105 installed packages compatible, including test/scanner tooling. No runtime dependency added. |
| Current Python dependency advisories | PASS | pip-audit 2.10.1, `python -m pip_audit --local --format json`: 105 installed packages checked against online PyPI advisories, **zero known vulnerabilities**. Cache-deserialization warnings caused entries to be ignored/refetched. Scope is this resolved environment, not every future version allowed by requirements or browser CDN libraries. |
| Current-file secret scan | PASS | detect-secrets 1.5.0, `scan --all-files --no-verify`, excluding `.venv`, `.git`, caches: **zero findings**. No credential verification or Git-history scan. `.env`/venvs ignored; no model keys configured in this session. |
| Scoped correctness/security review | PASS | Sentry/Ponytail/OWASP source review plus a separate reviewer. A custom-seed mismatch was corrected. A concurrent decision regression failed with two 200s before the fix and now requires one 200 and one 409. Reviewer confirmed the process-local lock resolves the finding within MemorySaver's scope. |
| Paid/live provider acceptance | BLOCKED | No GEMINI_API_KEY or GROQ_API_KEY was configured. No paid calls made; provider behavior covered by deterministic success/failure fixtures. Live Yahoo/SEC/Treasury feeds were not exercised by this change. |
| Authenticated PM role / tenant isolation | BLOCKED for production | Existing public paper-trading demo has no identities or tenants. Explicit boolean approval is a workflow control, not proof of an authorised human identity. |
| Durable/distributed decisions | BLOCKED for production | MemorySaver and the decision lock are process-local. Restarts lose paused runs. A durable checkpointer, transactional cross-process decision consumption and actor binding are required before real trading. |
| Model endpoint spend/abuse controls | BLOCKED for production | Existing public model endpoints lack deployment-verified identity, quotas and spend enforcement. No model endpoint exposure was added; new scenarios require no LLM calls. Do not connect unrestricted paid credentials on a public deployment. |
| AI trust boundary | PASS within changed flow | LLM stance may veto a BUY; deterministic risk and the explicit validated PM gate control allocation. The changed CIO prompt correctly calls P10/P90 an 80% model-implied interval. No new tools, prompts with execution privileges or model-selected allocations added. Full prompt-injection assessment not performed. |
| Backtest and load measurement | N/A to acceptance | Historical strategy tables were not rerun and no new performance claims made. Fixed scenario comparisons add two forecasts per forecast request; no representative production load/capacity test. |
| Infrastructure, migrations, recovery, broker execution | N/A | No infrastructure/schema migration or broker integration. No production deployment performed or assessed. |
| Landing-page preflight, light theme, Lighthouse | N/A | Existing dark research dashboard retained. No light theme or landing-page redesign; Lighthouse and full assistive-technology audit not run. |
| Full security audit | N/A | Focused source guidance and ordinary development tests only. Cloudflare's isolated exploit-test environment was unavailable; no complete audit, independent exploit confirmation or coverage claim. |

Runtime dependency ranges remain unlocked; future installs can resolve differently. LangGraph logs pre-existing warnings about application types not explicitly registered for future strict checkpoint deserialization. Before an enterprise release, the project owner must address the production blockers above, pin and test supported dependencies, and verify deployment controls. The existing demo's historical performance caveats remain applicable.

Ready to push for review as a paper-trading feature change. Production trading readiness is **BLOCKED**. Raw dependency/secret scan results, source manifest and browser screenshots were retained outside the repository in the local `apex-alpha-evidence` directory.
