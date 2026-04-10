-- Unified initialization script for MeshML shared database

-- data_batches table (Task Orchestrator)
CREATE TABLE IF NOT EXISTS data_batches (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    gcs_path TEXT,
    status TEXT NOT NULL,
    assigned_worker_id TEXT,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_data_batches_status ON data_batches(status);
CREATE INDEX IF NOT EXISTS idx_data_batches_job_id ON data_batches(job_id);

-- metrics table (Metrics Service)
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

-- checkpoint version column (Model Registry)
ALTER TABLE models
ADD COLUMN IF NOT EXISTS checkpoint_version INTEGER DEFAULT 0;
