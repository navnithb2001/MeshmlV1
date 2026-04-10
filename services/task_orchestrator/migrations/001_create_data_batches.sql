-- Create data_batches table for Task Orchestrator assignment engine
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
