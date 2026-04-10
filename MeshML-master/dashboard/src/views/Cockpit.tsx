import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, CheckCircle, Clock, XCircle, Loader2 } from 'lucide-react';
import { jobsAPI, modelsAPI } from '@/lib/api';
import { useToast } from '@/components/Toast';
import ConfirmModal from '@/components/ConfirmModal';
import clsx from 'clsx';

interface JobInfo {
  id: string;
  status: string;
  model_id: string | null;
  dataset_id: string | null;
  config: Record<string, any> | null;
  progress: { current_batch?: number; total_batches?: number; loss?: number; accuracy?: number } | null;
  created_at: string;
  completed_at: string | null;
}

export default function Cockpit() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<JobInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const toast = useToast();

  useEffect(() => {
    if (!id) return;
    const fetchJob = async () => {
      try {
        const data = await jobsAPI.getJob(id);
        setJob(data as any);
      } catch (err) {
        console.error('Failed to fetch job', err);
      } finally {
        setLoading(false);
      }
    };
    fetchJob();
    const interval = setInterval(fetchJob, 5000);
    return () => clearInterval(interval);
  }, [id]);

  const handleDownload = async () => {
    if (!job?.model_id) return;
    setDownloading(true);
    try {
      const res = await modelsAPI.getDownloadSignedUrl(String(job.model_id));
      if (res.download_url) {
        window.open(res.download_url, '_blank');
      } else {
        toast.warning('Download URL not available yet. The model may still be processing.');
      }
    } catch (err) {
      console.error('Download failed', err);
      toast.error('Failed to get download link. The model artifact may not be ready.');
    } finally {
      setDownloading(false);
    }
  };

  const handleCancel = async () => {
    if (!id) return;
    setDeleting(true);
    try {
      await jobsAPI.cancelJob(id);
      toast.success('Job cancelled successfully');
      navigate(-1);
    } catch (err) {
      console.error('Failed to cancel job', err);
      toast.error('Failed to cancel job');
    } finally {
      setDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const statusUpper = (job?.status || '').toUpperCase();
  const allowCancel = !['COMPLETED', 'FAILED', 'CANCELLED'].includes(statusUpper);
  const StatusIcon = statusUpper === 'COMPLETED' ? CheckCircle :
                     statusUpper === 'FAILED' || statusUpper === 'CANCELLED' ? XCircle :
                     statusUpper === 'PENDING' || statusUpper === 'WAITING' ? Clock :
                     Loader2;

  const statusColor = statusUpper === 'COMPLETED' ? 'text-emerald-500' :
                      statusUpper === 'FAILED' || statusUpper === 'CANCELLED' ? 'text-rose-500' :
                      statusUpper === 'PENDING' || statusUpper === 'WAITING' ? 'text-amber-500' :
                      'text-cyan-500';

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-4">
          <button onClick={() => navigate(-1)} className="p-2 border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors text-slate-500">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h1 className="text-xl font-semibold tracking-tight uppercase text-slate-900 dark:text-slate-50">
              Job Details
            </h1>
            <p className="text-sm font-mono text-slate-500 dark:text-slate-400 mt-1">
              {id}
            </p>
          </div>
        </div>
        <button
          onClick={() => setShowDeleteConfirm(true)}
          disabled={deleting || !allowCancel}
          className="flex items-center gap-2 px-4 py-2 border border-rose-200 dark:border-rose-900/50 text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-900/20 disabled:opacity-50 text-sm font-medium uppercase tracking-wider transition-colors"
        >
          {deleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <XCircle className="w-4 h-4" />}
          Cancel Job
        </button>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="w-8 h-8 text-slate-400 animate-spin" />
        </div>
      ) : job ? (
        <div className="flex-1 min-h-0 flex flex-col gap-4">
          {/* Status Panel */}
          <div className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 flex flex-wrap gap-8 items-center justify-between">
            <div className="space-y-1">
              <div className="text-xs uppercase tracking-widest text-slate-500">Status</div>
              <div className={clsx("font-mono text-lg font-bold uppercase flex items-center gap-2", statusColor)}>
                <StatusIcon className={clsx("w-5 h-5", statusUpper === 'ACTIVE' || statusUpper === 'TRAINING' ? 'animate-spin' : '')} />
                {job.status}
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-xs uppercase tracking-widest text-slate-500">Model ID</div>
              <div className="font-mono text-lg text-slate-900 dark:text-slate-50">{job.model_id || '---'}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs uppercase tracking-widest text-slate-500">Dataset ID</div>
              <div className="font-mono text-sm text-slate-900 dark:text-slate-50">{job.dataset_id || '---'}</div>
            </div>
            {job.config?.task_type && (
              <div className="space-y-1">
                <div className="text-xs uppercase tracking-widest text-slate-500">Task Type</div>
                <div className="font-mono text-lg text-slate-900 dark:text-slate-50 capitalize">{job.config.task_type}</div>
              </div>
            )}
            {job.progress && (
              <div className="space-y-1">
                <div className="text-xs uppercase tracking-widest text-slate-500">Batches</div>
                <div className="font-mono text-lg text-slate-900 dark:text-slate-50">
                  {job.progress.current_batch ?? 0} / {job.progress.total_batches ?? '?'}
                </div>
              </div>
            )}
            {job.progress && typeof job.progress.loss === 'number' && (
              <div className="space-y-1">
                <div className="text-xs uppercase tracking-widest text-slate-500">Current Loss</div>
                <div className="font-mono text-lg text-amber-600 dark:text-amber-500">
                  {job.progress.loss.toFixed(4)}
                </div>
              </div>
            )}
            {job.progress && typeof job.progress.accuracy === 'number' && (
              <div className="space-y-1">
                <div className="text-xs uppercase tracking-widest text-slate-500">Training Acc</div>
                <div className="font-mono text-lg text-emerald-600 dark:text-emerald-500">
                  {(job.progress.accuracy * 100).toFixed(2)}%
                </div>
              </div>
            )}
            <div className="space-y-1">
              <div className="text-xs uppercase tracking-widest text-slate-500">Created</div>
              <div className="font-mono text-sm text-slate-900 dark:text-slate-50">
                {new Date(job.created_at).toLocaleString()}
              </div>
            </div>
          </div>

          {/* Download Model Panel */}
          <div className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-8 flex flex-col items-center justify-center gap-6">
            <div className="text-sm font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-50">
              Model Artifact
            </div>
            {job.model_id ? (
              <>
                <p className="text-sm text-slate-500 text-center max-w-md">
                  {statusUpper === 'COMPLETED'
                    ? 'Training complete. Download the final trained model artifact.'
                    : 'Training is in progress. You can attempt to download the latest checkpoint.'}
                </p>
                <button
                  onClick={handleDownload}
                  disabled={downloading}
                  className={clsx(
                    "flex items-center gap-3 px-8 py-3 text-sm font-semibold uppercase tracking-wider transition-all",
                    statusUpper === 'COMPLETED'
                      ? "bg-emerald-600 hover:bg-emerald-700 text-white"
                      : "bg-cyan-600 hover:bg-cyan-700 text-white",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  {downloading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Download className="w-5 h-5" />
                  )}
                  {downloading ? 'Preparing...' : 'Download Model'}
                </button>
              </>
            ) : (
              <p className="text-sm text-slate-500 font-mono">NO_MODEL_ATTACHED</p>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-slate-500 font-mono text-sm">
          JOB_NOT_FOUND
        </div>
      )}

      <ConfirmModal
        isOpen={showDeleteConfirm}
        title="Cancel Job"
        message="Are you sure you want to cancel this job? This action will terminate training and cannot be undone."
        detail={id}
        confirmLabel="Cancel Job"
        onCancel={() => setShowDeleteConfirm(false)}
        onConfirm={handleCancel}
      />
    </div>
  );
}
