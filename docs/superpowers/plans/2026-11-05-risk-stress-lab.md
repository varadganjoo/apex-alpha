# Risk stress lab implementation plan

**Goal:** Let researchers see how fragile a position size is to return and volatility assumptions, with measured simulated tail losses and explicit human decisions.

**Architecture:** Reuse the seeded forecaster and risk guard. Add three fixed comparisons to the forecast response and risk desk; share them with MCP. Keep all execution simulated.

**Stack:** Existing Python, NumPy, Pydantic, FastAPI, LangGraph and vanilla HTML/JavaScript. No new runtime dependency.

**Acceptance criteria:**
- Compute 95% VaR and expected shortfall from each horizon's simulated return distribution, not scaled P10 proxies. Distinguish arithmetic mean return from median return.
- Compare base, CAPM without the reversal tilt, and 1.5x volatility with the same random seed and risk-free input. Clearly identify hypothetical assumptions, price provenance, and model uncertainty.
- Every proposal pauses. Approval must be explicit; a resize may only decrease the proposed position and must remain finite and within 15%. HOLD proposals carry zero target capital.
- Show the comparisons in an accessible, horizontally scrollable risk-desk table. Preserve the terminal's existing visual language.

**Scope:** schemas, quant engine, risk guard, graph gate/commit, forecast API, MCP tools, risk-desk UI, associated tests, README and methodology. No broker integration, performance claims, new trading signals, accounts, or persistence system.

**Trust boundaries:** External market data and LLM text are untrusted. Scenarios are deterministic research outputs; models cannot approve or increase allocations. This remains a public, unauthenticated paper-trading demo; production identity, distributed state and paid-endpoint abuse controls are outside this change.

## Implementation

- [x] Add distribution fixtures and negative approval tests; run them against the original behavior to verify failures.
- [x] Implement distribution tail statistics and three shared scenarios; verify numerical and cross-interface behavior.
- [x] Enforce explicit approval and reduction-only sizing at the graph's final decision point and HTTP boundary; verify rejection, invalid inputs and replay.
- [x] Render and test the risk desk at desktop/mobile widths with keyboard and reduced-motion checks; document assumptions and limits.
- [x] Run the full suite, dependency and secret checks, review connected callers, record release evidence.
- [x] Commit and push the tested branch; open [pull request #1](https://github.com/varadganjoo/apex-alpha/pull/1). [GitHub CI](https://github.com/varadganjoo/apex-alpha/actions/runs/42146601040) passed on Python 3.11 and 3.12.

**Review focus:** all-positive simulated returns; tiny tail samples; non-finite overrides; direct graph callers bypassing HTTP validation; sample data mistaken for a live forecast. Each needs a regression or rendered journey.

**Procedures:** Enterprise router; Minimalist MVP for scope; Ponytail full; Superpowers design/planning/TDD/verification; Spellbook FastAPI/API/accessibility; LangGraph fundamentals/HITL; OWASP code/secrets/dependency review; Sentry and Ponytail change review; Anthropic rendered testing. Cloudflare guidance is source-only here: this Windows environment does not establish its isolated exploit-test requirements. Full audit, landing-page redesign, infrastructure/migration reviews and production deployment are N/A to this bounded feature change.
