# Interview Prep Workbook

Fill the bracketed numbers with your real results. Don't use figures you can't defend.

## STAR stories

**1. Fraud model (XGBoost, XMmobile orders)**
- **S/T:** Needed to flag fraudulent mobile orders from [N] orders/month with ~[X]% fraud rate.
- **A:** Engineered risk features (country, email-domain, BYOD item type), used class weighting, evaluated with PR-AUC and recall at fixed precision.
- **R:** [metric improvement / fraud caught / review workload reduced].
- **Be ready for:** leakage, threshold choice, drift, retraining cadence.

**2. Trino memory failure (debugging at scale)**
- **S/T:** A 12-month extract hit the 15 GB per-node limit (`EXCEEDED_LOCAL_MEMORY_LIMIT`).
- **A:** Read the error: the biggest consumer was the table scan (`a.*`, wide rows). Partitioned the extract into monthly slices with identical filters. Next: column pruning, pushdown, aggregating in the warehouse.
- **R:** [query completes; runtime/memory numbers].

**3. Data quality / reporting ownership**
- **S/T:** [report/pipeline that depended on trustworthy data].
- **A:** [validation checks, monitoring, documentation].
- **R:** [incidents prevented / time saved].

## Question bank

**SQL / performance:** window functions; `EXPLAIN` and partition pruning; join order and skew; why `select *` is costly in columnar engines.
**Pipelines:** idempotency; backfills; retries and alerting; schema evolution; exactly-once vs at-least-once.
**ML:** imbalanced metrics; time-based validation; training/serving skew; drift (PSI, KS); when to retrain; shadow deployment and A/B tests.
**LLM/RAG:** chunking and embeddings; retrieval evaluation (recall@k); hallucination controls; RAG vs fine-tuning; agent tool-calling safety.
**System design:** "Design a near-real-time fraud scoring system": ingest, feature store, model serving, monitoring, rollback, cost.
**Telecom (preferred):** capacity management, network deployment data, KPIs; connect fraud/order data to telecom operations.

## Gap-closing checklist

- [ ] Run the PySpark job on Databricks Community Edition and write Delta
- [ ] Run the DAG in Airflow via Docker
- [ ] Add MLflow model registry and a promotion gate to the pipeline
- [ ] Build a Tableau or Power BI view of `artifacts/` outputs
- [ ] Build a small RAG app (embeddings, vector search, retrieval eval) in `portfolio/`
- [ ] Practise one 30-minute system-design walkthrough out loud
