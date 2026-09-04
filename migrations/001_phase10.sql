-- Phase 10 PostgreSQL schema. Apply with a migration runner before starting production.
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE,
    customer_id VARCHAR(64) NOT NULL,
    merchant_id VARCHAR(64) NOT NULL,
    amount NUMERIC(18, 2) NOT NULL CHECK (amount > 0),
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    merchant_category VARCHAR(50) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    country VARCHAR(10) NOT NULL,
    raw_payload_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    version VARCHAR(64) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL,
    artifact_fingerprint VARCHAR(128),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE REFERENCES transactions(transaction_id),
    model_version VARCHAR(64) NOT NULL REFERENCES model_versions(version),
    fraud_probability DOUBLE PRECISION NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    ml_risk_score INTEGER NOT NULL CHECK (ml_risk_score BETWEEN 0 AND 100),
    final_risk_score INTEGER NOT NULL CHECK (final_risk_score BETWEEN 0 AND 100),
    anomaly_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    risk_level VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    triggered_rules_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    fallback_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_factors (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE REFERENCES risk_predictions(transaction_id),
    risk_factors_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    protective_factors_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    llm_explanation_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS verification_actions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(transaction_id),
    previous_decision VARCHAR(20) NOT NULL,
    analyst_action VARCHAR(50) NOT NULL,
    analyst_notes TEXT,
    analyst_id VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    transaction_id VARCHAR(64) REFERENCES transactions(transaction_id),
    model_version VARCHAR(64) NOT NULL,
    fraud_probability DOUBLE PRECISION,
    risk_score INTEGER,
    decision VARCHAR(20),
    top_risk_factors JSONB NOT NULL DEFAULT '[]'::jsonb,
    action VARCHAR(50) NOT NULL,
    actor VARCHAR(64) NOT NULL,
    reason TEXT NOT NULL,
    request_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    action_taken VARCHAR(50) NOT NULL,
    details_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_transactions_customer_created ON transactions(customer_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_risk_predictions_decision_created ON risk_predictions(decision, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_risk_predictions_risk_level_created ON risk_predictions(risk_level, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_verification_actions_transaction_created ON verification_actions(transaction_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_audit_logs_transaction_timestamp ON audit_logs(transaction_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS ix_audit_logs_request_id ON audit_logs(request_id);
CREATE INDEX IF NOT EXISTS ix_audit_logs_decision_timestamp ON audit_logs(decision, timestamp DESC);
