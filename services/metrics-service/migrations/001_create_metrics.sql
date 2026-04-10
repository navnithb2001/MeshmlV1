CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    step INTEGER NOT NULL,
    loss DOUBLE PRECISION,
    accuracy DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_job_id ON metrics (job_id);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics (step);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp);
