**To:** Aisha<br>
**From:** Sandra<br>
**Subject:** FinCorp RAG Pipeline: LCEL Architecture Proposal (Parallel Retrieval, Router Chains, Pydantic Contracts)<br>

Hi Aisha,

Below is my proposed architecture for the FinCorp multi-source workflow, with the reasoning behind it. A working proof-of-concept stub (`workflow_architecture.py`, GPT-4o-mini) implements everything described here.

## 1. Summary

FinCorp needs two very different kinds of data: **structured financial records** and **unstructured market news**. Their current system struggles with hallucinated values and noisy LLM output. I propose three LCEL building blocks:

1. **A parallel data loader** (`RunnableParallel`) that retrieves both sources concurrently.
2. **A router chain** (`RunnableLambda`) that sends each query to a specialist inference chain based on its intent.
3. **Pydantic output parsers** that turn every LLM response into a validated object, so downstream reporting tools only ever receive data that matches a defined schema.

## 2. Architecture

```
user input {query, intent}
      |
      v
RunnableParallel (concurrent retrieval)
  |-- fetch_structured_financials(query)  --> financial_data
  '-- fetch_unstructured_news(query)      --> news_data
      |
      v
Router (RunnableLambda, reads intent)
  |-- "financial" --> financial_prompt | gpt-4o-mini | PydanticOutputParser(FinancialSummary)
  |-- "news"      --> news_prompt      | gpt-4o-mini | PydanticOutputParser(MarketNews)
  '-- other       --> explicit error result (no LLM call)
      |
      v
validated Pydantic object --> downstream reporting tools
```

Each stage is a Runnable, so the whole pipeline is one composable object with the same `.invoke()`, `.batch()`, and `.stream()` interface.

## 3. Parallel execution

The two retrieval functions are independent: neither needs the other's output. Running them sequentially would make total latency the **sum** of both calls. With `RunnableParallel`, both run concurrently and total latency is roughly the **slowest** call.

- **Benefit:** lower end-to-end latency, which matters most when the real sources are a database query and a news API.
- **Benefit:** adding a third source (e.g., analyst reports) is one new dictionary entry, with no change to the downstream steps.
- **Cost to watch:** the PoC fetches both sources for every request, even though a given intent uses only one. For the PoC this is acceptable. In production, if the sources are slow or billed per call, we should route first and fetch only what the chosen branch needs. That is an easy change because both pieces are already separate Runnables.

## 4. Linear extended chains vs. router chains

| Concern | Linear extended chain | Router chain (proposed) |
|---|---|---|
| Structure | One fixed path through every step | Classifies the request, then runs only the relevant branch |
| Prompts | One prompt must cover every case, which grows long and ambiguous | Each branch has a focused prompt (financial analyst vs. news analyst) |
| Output schema | One generic schema, or optional fields filled with guesses | One strict schema per branch (`FinancialSummary`, `MarketNews`) |
| Latency and cost | Pays for every step on every request | Pays only for the selected branch |
| Failure isolation | A failure in any step breaks the whole request | A failure stays inside one branch, so it is easier to test, trace, and fix |
| Extensibility | Adding a case means rewriting the shared prompt and schema, with regression risk | Adding a case means adding a branch (prompt, parser, route entry) |
| Simplicity | Simplest to read for a single, uniform task | Slightly more moving parts; needs a reliable intent signal |

**Recommendation:** use a router for FinCorp. Their two data types need different prompts and different output contracts, which is exactly the case a linear chain handles badly. A linear chain stays the better choice for steps that truly depend on each other (e.g., summarize, then translate), and we can still use one *inside* any branch.

**Trade-offs to be aware of:**
- **Routing quality becomes a dependency.** A wrong intent sends the query to the wrong branch. In the PoC the intent is supplied in the input. In production I propose a low-temperature LLM classifier whose output is itself a Pydantic `Literal["financial", "news"]`, so an invalid label fails fast instead of silently misrouting.
- **Unknown intents must be handled explicitly.** The router returns a structured error for unsupported intents, so no LLM call is made and nothing malformed reaches downstream tools.

## 5. How Pydantic contracts prevent noise and hallucinated structure

Each branch ends in a `PydanticOutputParser` tied to a schema. This provides four guarantees:

1. **The prompt carries the contract.** `get_format_instructions()` injects the schema (field names, types, and our field descriptions) into the prompt, so the model is told exactly what shape to return.
2. **The parser rejects non-conforming output.** If the model returns extra prose, a missing field, or a wrong type (e.g., a string where `total_revenue: float` is required), parsing raises an error. Malformed output does not reach the next stage.
3. **Fields are constrained and documented.** For example, `quarter` must follow the `Q3 2023` format, `total_revenue` and `net_income` are floats in millions, `sentiment_score` is a float between -1.0 and 1.0, and `top_headlines` contains exactly three items. The reporting tools can rely on those rules.
4. **Models are required, not optional.** Fields use `Field(...)` so the model cannot quietly omit values. A missing field is a visible failure and not a silent gap.

**Important limit:** Pydantic enforces *structure and types*, not *truth*. It cannot prove that a revenue number is correct. We reduce hallucinated *values* through two other measures:
- **Grounding:** each prompt includes only the retrieved record or news text, and instructs the model to summarize that data. The mock sources are built so every required field exists in the source text, so the model has no reason to invent values.
- **Failure handling:** when validation fails, we can retry once with LangChain's retry or fixing parsers, then return an error. We never pass a partial object downstream. For critical numeric fields, I recommend a later check comparing parsed values against the source record.

Together, schema, grounding, and fail-fast validation keep noise from propagating from one stage to the next.

## 6. PoC scope and next steps

**Implemented in the stub:** concurrent retrieval with `RunnableParallel`, intent-based routing in a `RunnableLambda`, two prompt/parser chains on GPT-4o-mini, and Pydantic models for both outputs, wired into one pipeline with the `|` operator.

**Proposed next steps:**
1. Replace the mock retrieval functions with the real financial database and news API.
2. Add the LLM intent classifier with a `Literal` output schema.
3. Add retry/fix handling for parser failures, plus logging and tracing (e.g., LangSmith) on every branch.
4. Add tests for each branch and for the unknown-intent path.
5. Decide whether to route before retrieval, based on measured source latency and cost.

Happy to walk through the stub or adjust the approach. Please let me know if you have any questions.

Best regards,<br>
Sandra