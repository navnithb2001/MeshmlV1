import { useState, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { X, UploadCloud, Code, Settings, Play, CheckCircle, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { jobsAPI, datasetsAPI, modelsAPI } from '@/lib/api';
import { useToast } from '@/components/Toast';
import clsx from 'clsx';
import { useQuery } from '@tanstack/react-query';

interface SetupModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function getErrorMessage(err: unknown, fallback: string): string {
  const candidate = err as {
    response?: { data?: { detail?: unknown; message?: unknown } };
  };
  const detail = candidate?.response?.data?.detail;
  const message = candidate?.response?.data?.message;

  if (Array.isArray(detail)) {
    const joined = detail
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg?: unknown }).msg ?? '');
        }
        return '';
      })
      .filter(Boolean)
      .join(' | ');
    if (joined) return joined;
  }

  if (typeof detail === 'string' && detail.trim()) return detail;
  if (typeof message === 'string' && message.trim()) return message;
  return fallback;
}

export default function SetupModal({ isOpen, onClose }: SetupModalProps) {
  const navigate = useNavigate();
  const toast = useToast();
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [targetVersion, setTargetVersion] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [shardingStrategy, setShardingStrategy] = useState('stratified');

  // File Upload State
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetMode, setDatasetMode] = useState<'upload' | 'existing'>('upload');
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState('');
  const [datasetId, setDatasetId] = useState<string | null>(null);

  const { groupId } = useParams<{ groupId: string }>();

  const datasetInputRef = useRef<HTMLInputElement>(null);
  const modelInputRef = useRef<HTMLInputElement>(null);

  const { data: datasets } = useQuery({
    queryKey: ['setup-modal-datasets'],
    queryFn: () => datasetsAPI.listDatasets(),
    enabled: isOpen,
  });

  if (!isOpen) return null;

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDatasetDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setDatasetFile(e.dataTransfer.files[0]);
    }
  };

  const handleModelDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setModelFile(e.dataTransfer.files[0]);
    }
  };

  const handleDataStepSubmit = async () => {
    if (datasetMode === 'existing') {
      if (!selectedDatasetId) {
        toast.warning('Select an existing dataset first.');
        return;
      }
      const selected = (datasets?.datasets || []).find((d) => d.id === selectedDatasetId);
      const selectedStatus = (selected?.status || '').toLowerCase();
      if (selected && !['available', 'uploaded'].includes(selectedStatus)) {
        toast.warning(`Dataset is ${selected.status}. Choose one that is available.`);
        return;
      }
      setDatasetId(selectedDatasetId);
      setStep(2);
      return;
    }

    if (!datasetFile) {
      toast.warning('Select a dataset file before continuing.');
      return;
    }
    // Fast Upload: We defer actual file uploading until Start Job
    setStep(2);
  };

  const handleCodeStepSubmit = async () => {
    if (!groupId) {
      toast.error('No Group Context found.');
      return;
    }
    if (!modelFile) {
      toast.warning('Upload a model file before continuing.');
      return;
    }
    if (!modelName.trim()) {
      toast.warning('Model Name is required when uploading architecture.');
      return;
    }
    setLoading(true);
    try {
      await modelsAPI.uploadModelArchitecture(modelFile, modelName, groupId);
      setStep(3);
      toast.success('Model architecture uploaded.');
    } catch (err) {
      console.error('Failed to upload model architecture', err);
      toast.error(getErrorMessage(err, 'Failed to upload model architecture.'));
    } finally {
      setLoading(false);
    }
  };

  const handleStartJob = async () => {
    if (!groupId) {
      toast.error('No Group Context found.');
      return;
    }
    const parsedTarget = targetVersion.trim() === '' ? 1000 : Number(targetVersion);
    if (!Number.isFinite(parsedTarget) || parsedTarget <= 0) {
      toast.warning('Convergence target must be a number greater than 0.');
      return;
    }
    setLoading(true);
    try {
      let finalDatasetId = datasetId;

      if (datasetMode === 'upload' && datasetFile) {
        toast.info('Uploading and validating dataset...');
        // Fast Upload pattern: dataset triggers background extraction
        const res = await datasetsAPI.uploadDataset(
          [datasetFile], 
          shardingStrategy,
          undefined,
          undefined,
          (progressEvent) => {
            if (progressEvent.total) {
              const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              setUploadProgress(percent);
            }
          }
        );
        finalDatasetId = res.dataset_id;
      }

      // POST /api/jobs (JobCreateRequest payload)
      const job = await jobsAPI.createJob({ 
        group_id: groupId,
        dataset_id: finalDatasetId || undefined,
        config: { 
          final_version: Math.floor(parsedTarget),
          shard_strategy: shardingStrategy
        }
      });
      // Immediately routes user to Live Dashboard
      toast.success('Training run started successfully.');
      setUploadProgress(0);
      onClose();
      navigate(`/jobs/${job.id}/live`);
    } catch (err) {
      console.error('Failed to start job', err);
      toast.error(getErrorMessage(err, 'Failed to start training run.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm">
      <div className="w-full max-w-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 p-4 shrink-0">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-50">
            New Training Run
          </h2>
          <button onClick={onClose} className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-500">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 flex-1 max-h-[70vh] overflow-y-auto">
          {/* Step 1: Data */}
          {step === 1 && (
            <div className="space-y-4 animate-in fade-in zoom-in duration-200">
              <div className="text-xs font-mono uppercase text-slate-500 mb-2">Step 1/3: Data Ingestion</div>
              <div className="flex gap-2">
                <button
                  onClick={() => setDatasetMode('upload')}
                  className={clsx(
                    "text-xs font-medium uppercase tracking-wider px-3 py-2 border transition-colors",
                    datasetMode === 'upload'
                      ? "border-cyan-600 text-cyan-600 dark:border-cyan-400 dark:text-cyan-400"
                      : "border-slate-300 dark:border-slate-700 text-slate-500"
                  )}
                >
                  Upload New
                </button>
                <button
                  onClick={() => setDatasetMode('existing')}
                  className={clsx(
                    "text-xs font-medium uppercase tracking-wider px-3 py-2 border transition-colors",
                    datasetMode === 'existing'
                      ? "border-cyan-600 text-cyan-600 dark:border-cyan-400 dark:text-cyan-400"
                      : "border-slate-300 dark:border-slate-700 text-slate-500"
                  )}
                >
                  Use Existing
                </button>
              </div>

              {datasetMode === 'existing' && (
                <div className="border border-slate-200 dark:border-slate-800 p-4 max-h-72 overflow-y-auto">
                  {(!datasets?.datasets || datasets.datasets.length === 0) && (
                    <div className="text-sm font-mono text-slate-500">No previous datasets found.</div>
                  )}
                  <div className="space-y-2">
                    {(datasets?.datasets || []).map((dataset) => (
                      <button
                        key={dataset.id}
                        onClick={() => setSelectedDatasetId(dataset.id)}
                        className={clsx(
                          "w-full text-left border p-3 transition-colors",
                          selectedDatasetId === dataset.id
                            ? "border-cyan-600 bg-cyan-50/40 dark:bg-cyan-950/20"
                            : "border-slate-200 dark:border-slate-800 hover:border-cyan-500"
                        )}
                      >
                        <div className="text-sm font-medium text-slate-900 dark:text-slate-50">{dataset.name}</div>
                        <div className="text-xs font-mono text-slate-500 mt-1">{dataset.id}</div>
                        <div className="text-xs text-slate-500 mt-1 uppercase">{dataset.status}</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {datasetMode === 'upload' && (
              <div 
                onDragOver={handleDragOver}
                onDrop={handleDatasetDrop}
                onClick={() => datasetInputRef.current?.click()}
                className={clsx(
                  "border border-dashed p-12 flex flex-col items-center justify-center cursor-pointer transition-colors group",
                  datasetFile 
                    ? "border-emerald-500 bg-emerald-50/10 dark:bg-emerald-950/20" 
                    : "border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 hover:border-cyan-500"
                )}
              >
                <input
                  ref={datasetInputRef}
                  type="file"
                  accept=".zip,.tar,.gz,.tar.gz"
                  className="hidden"
                  onChange={(e) => { if (e.target.files?.[0]) setDatasetFile(e.target.files[0]); }}
                />
                {datasetFile ? (
                  <>
                    <CheckCircle className="w-8 h-8 text-emerald-500 mb-4" />
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-50">{datasetFile.name}</p>
                    <p className="text-xs text-slate-500 font-mono mt-2">{(datasetFile.size / 1024 / 1024).toFixed(2)} MB</p>
                  </>
                ) : (
                  <>
                    <UploadCloud className="w-8 h-8 text-slate-400 group-hover:text-cyan-500 mb-4 transition-colors" />
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-50">Drag & drop dataset here</p>
                    <p className="text-xs text-slate-500 font-mono mt-2">
                      Supported: .zip, .tar, .tar.gz (imagefolder, csv, coco is auto-detected)
                    </p>
                  </>
                )}
              </div>
              )}
              <div className="flex justify-end pt-4">
                <button 
                  onClick={handleDataStepSubmit} 
                  disabled={loading}
                  className="bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-900 text-sm font-medium py-2 px-6 disabled:opacity-50"
                >
                  {loading ? 'UPLOADING...' : 'NEXT'}
                </button>
              </div>
            </div>
          )}

          {/* Step 2: Code */}
          {step === 2 && (
            <div className="space-y-4 animate-in fade-in zoom-in duration-200">
              <div className="flex justify-between items-center mb-2">
                <div className="text-xs font-mono uppercase text-slate-500">Step 2/3: Model Architecture</div>
                <button
                  onClick={() => setShowHelpModal(true)}
                  className="text-xs font-medium uppercase tracking-wider px-3 py-1.5 border border-amber-600 text-amber-600 dark:border-amber-500 dark:text-amber-500 hover:bg-amber-50 dark:hover:bg-amber-950/20 transition-colors flex items-center gap-1.5"
                >
                  <Info className="w-3.5 h-3.5" />
                  Code Format Guide
                </button>
              </div>
              
              <div className="space-y-2">
                <label className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Model Reference Name
                </label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm placeholder:text-slate-400 focus:outline-none focus:border-cyan-600 dark:focus:border-cyan-400 transition-colors"
                  placeholder="e.g. gpt2-base-architecture"
                />
              </div>

              <div 
                onDragOver={handleDragOver}
                onDrop={handleModelDrop}
                onClick={() => modelInputRef.current?.click()}
                className={clsx(
                  "border border-dashed p-12 flex flex-col items-center justify-center cursor-pointer transition-colors group",
                  modelFile 
                    ? "border-emerald-500 bg-emerald-50/10 dark:bg-emerald-950/20" 
                    : "border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 hover:border-cyan-500"
                )}
              >
                <input
                  ref={modelInputRef}
                  type="file"
                  accept=".py"
                  className="hidden"
                  onChange={(e) => { if (e.target.files?.[0]) setModelFile(e.target.files[0]); }}
                />
                {modelFile ? (
                  <>
                    <CheckCircle className="w-8 h-8 text-emerald-500 mb-4" />
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-50">{modelFile.name}</p>
                    <p className="text-xs text-slate-500 font-mono mt-2">{(modelFile.size / 1024).toFixed(2)} KB</p>
                  </>
                ) : (
                  <>
                    <Code className="w-8 h-8 text-slate-400 group-hover:text-cyan-500 mb-4 transition-colors" />
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-50">Drag & drop model code here (.py)</p>
                  </>
                )}
              </div>
              <div className="flex justify-between pt-4">
                <button onClick={() => setStep(1)} disabled={loading} className="border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-slate-50 text-sm font-medium py-2 px-6 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
                  BACK
                </button>
                <button onClick={handleCodeStepSubmit} disabled={loading} className="bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-900 text-sm font-medium py-2 px-6 disabled:opacity-50">
                   {loading ? 'UPLOADING...' : 'NEXT'}
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Config */}
          {step === 3 && (
            <div className="space-y-4 animate-in fade-in zoom-in duration-200">
              <div className="text-xs font-mono uppercase text-slate-500 mb-2">Step 3/3: Configuration</div>
              <div className="border border-slate-200 dark:border-slate-800 p-6">
                <label className="flex items-center space-x-2 text-sm font-medium text-slate-900 dark:text-slate-50 mb-4 uppercase tracking-wider">
                  <Settings className="w-4 h-4 text-slate-500" />
                  <span>Convergence Target</span>
                </label>
                <div className="space-y-2">
                  <label className="text-xs font-mono text-slate-500">final_version (epochs/steps)</label>
                  <input 
                    type="number" 
                    value={targetVersion}
                    onChange={(e) => setTargetVersion(e.target.value)}
                    placeholder="e.g. 10000"
                    className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm focus:outline-none focus:border-cyan-600 dark:focus:border-cyan-400 transition-colors"
                  />
                </div>
                
                {/* Advanced Configuration Accordion */}
                <div className="mt-6 border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50">
                  <button 
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)} 
                    className="w-full flex items-center justify-between p-4 text-sm font-medium text-slate-900 dark:text-slate-50 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors"
                  >
                    <span className="uppercase tracking-wider">Advanced Configuration</span>
                    {showAdvanced ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
                  </button>
                  
                  {showAdvanced && (
                    <div className="p-4 border-t border-slate-200 dark:border-slate-800 space-y-4">
                      
                      {/* Sharding Strategy */}
                      <div className="space-y-2">
                        <label className="text-xs font-mono text-slate-500 uppercase">Data Sharding Strategy</label>
                        <div className="space-y-2">
                          {[
                            { value: 'stratified', label: 'Stratified', tooltip: 'Default. Maintains the global class distribution across all shards evenly.' },
                            { value: 'random', label: 'Random (IID)', tooltip: 'Purely random Independent and Identically Distributed chunks.' },
                            { value: 'non_iid', label: 'Non-IID', tooltip: 'Simulates federated learning by skewing class distribution per shard using a Dirichlet distribution.' },
                            { value: 'sequential', label: 'Sequential', tooltip: 'Grabs contiguous blocks of data without shuffling. Mostly for debugging.' },
                          ].map((option) => (
                            <div 
                              key={option.value}
                              onClick={() => setShardingStrategy(option.value)}
                              className={clsx(
                                "relative group flex items-center justify-between p-3 border cursor-pointer transition-colors",
                                shardingStrategy === option.value 
                                  ? "border-cyan-500 bg-cyan-500/10" 
                                  : "border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-white dark:bg-slate-900"
                              )}
                            >
                              <span className={clsx(
                                "text-sm flex-1 font-medium",
                                shardingStrategy === option.value ? "text-cyan-600 dark:text-cyan-400" : "text-slate-700 dark:text-slate-300"
                              )}>
                                {option.label}
                              </span>
                              
                              <div className="relative flex items-center shrink-0">
                                <Info className="w-4 h-4 text-slate-400 dark:text-slate-500" />
                                
                                {/* Custom CSS Tooltip */}
                                <div className="absolute opacity-0 group-hover:opacity-100 transition-opacity duration-200 bottom-full right-0 mb-2 w-56 p-2 bg-slate-900 text-slate-50 text-xs rounded shadow-lg pointer-events-none z-10 whitespace-normal text-left">
                                  {option.tooltip}
                                  {/* Tooltip Arrow */}
                                  <div className="absolute top-full right-1 border-[5px] border-transparent border-t-slate-900" />
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                    </div>
                  )}
                </div>
              </div>
              
              <div className="flex justify-between pt-4">
                <button onClick={() => setStep(2)} disabled={loading} className="border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-slate-50 text-sm font-medium py-2 px-6 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors disabled:opacity-50">
                  BACK
                </button>
                <button onClick={handleStartJob} disabled={loading} className="bg-cyan-600 hover:bg-cyan-700 text-white text-sm font-medium py-2 px-6 flex items-center space-x-2 transition-colors disabled:opacity-50">
                  <Play className="w-4 h-4 fill-current" />
                  <span>
                    {loading 
                      ? (uploadProgress > 0 && uploadProgress < 100 ? `UPLOADING (${uploadProgress}%)...` : 'STARTING...') 
                      : 'START JOB'}
                  </span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Code Format Help Modal (Overlay) */}
      {showHelpModal && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/60 backdrop-blur-md p-4 animate-in fade-in">
          <div className="w-full max-w-4xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col max-h-[90vh]">
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 p-4 shrink-0 bg-slate-50 dark:bg-slate-950">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-50 flex items-center gap-2">
                <Code className="w-4 h-4 text-amber-500" />
                Required Python Model Format
              </h2>
              <button onClick={() => setShowHelpModal(false)} className="p-1 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors text-slate-500">
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto space-y-6">
              <p className="text-sm text-slate-700 dark:text-slate-300">
                MeshML requires your PyTorch <code className="bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-xs">model.py</code> file to globally export two things: a <code className="bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-xs">MODEL_METADATA</code> dictionary and a <code className="bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-xs">create_model()</code> factory function.
              </p>

              {/* ---- Parameter Reference ---- */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-50 mb-3">MODEL_METADATA Parameter Reference</h3>
                <div className="border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden text-xs">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-slate-100 dark:bg-slate-800/60 text-left">
                        <th className="px-3 py-2 font-semibold text-slate-700 dark:text-slate-300 w-[140px]">Parameter</th>
                        <th className="px-3 py-2 font-semibold text-slate-700 dark:text-slate-300">Description</th>
                        <th className="px-3 py-2 font-semibold text-slate-700 dark:text-slate-300 w-[180px]">Accepted Values</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800 text-slate-600 dark:text-slate-400">
                      <tr><td className="px-3 py-2 font-mono text-cyan-600 dark:text-cyan-400">name</td><td className="px-3 py-2">A human-readable identifier for your model. Used in the dashboard and logs.</td><td className="px-3 py-2 font-mono">any string</td></tr>
                      <tr><td className="px-3 py-2 font-mono text-cyan-600 dark:text-cyan-400">version</td><td className="px-3 py-2">Semantic version tag. Helps track model iterations across training runs.</td><td className="px-3 py-2 font-mono">"1.0", "2.1", etc.</td></tr>
                      <tr><td className="px-3 py-2 font-mono text-cyan-600 dark:text-cyan-400">framework</td><td className="px-3 py-2">The ML framework used. Currently only PyTorch is supported.</td><td className="px-3 py-2 font-mono">"pytorch"</td></tr>
                      <tr><td className="px-3 py-2 font-mono text-cyan-600 dark:text-cyan-400">input_shape</td><td className="px-3 py-2">Shape of a single input sample <strong>without</strong> the batch dimension. Workers use this to validate data shards.</td><td className="px-3 py-2 font-mono">[C, H, W] or [features]</td></tr>
                      <tr><td className="px-3 py-2 font-mono text-cyan-600 dark:text-cyan-400">output_shape</td><td className="px-3 py-2">Shape of the model's output tensor. Typically [num_classes] for classifiers or [1] for regressors.</td><td className="px-3 py-2 font-mono">[N] list of ints</td></tr>
                      <tr className="bg-amber-50/50 dark:bg-amber-950/10"><td className="px-3 py-2 font-mono text-amber-600 dark:text-amber-400">task_type</td><td className="px-3 py-2">Determines the training loop behavior — how loss is computed and which metrics are tracked.</td><td className="px-3 py-2 font-mono">"classification" | "regression" | "binary"</td></tr>
                      <tr className="bg-amber-50/50 dark:bg-amber-950/10"><td className="px-3 py-2 font-mono text-amber-600 dark:text-amber-400">loss</td><td className="px-3 py-2">The loss function the worker will use. Must match the task type (e.g., cross_entropy for classification, mse for regression).</td><td className="px-3 py-2 font-mono">"cross_entropy" | "mse" | "mae" | "bce" | "bce_with_logits"</td></tr>
                      <tr className="bg-amber-50/50 dark:bg-amber-950/10"><td className="px-3 py-2 font-mono text-amber-600 dark:text-amber-400">metrics</td><td className="px-3 py-2">List of evaluation metrics computed after each batch/epoch and streamed to the dashboard.</td><td className="px-3 py-2 font-mono">["accuracy"] | ["mse", "r2"] | ["accuracy", "f1"]</td></tr>
                      <tr><td className="px-3 py-2 font-mono text-cyan-600 dark:text-cyan-400">num_outputs</td><td className="px-3 py-2">Number of output neurons. For classification = num_classes; for regression = 1 (or target dims).</td><td className="px-3 py-2 font-mono">positive integer</td></tr>
                      <tr><td className="px-3 py-2 font-mono text-cyan-600 dark:text-cyan-400">target_dtype</td><td className="px-3 py-2">PyTorch dtype for labels. Classification uses "long" (integer class IDs); regression uses "float".</td><td className="px-3 py-2 font-mono">"long" | "float"</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* ---- Task Type Quick Reference ---- */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-50 mb-3">Task Type Quick Reference</h3>
                <div className="grid grid-cols-3 gap-3 text-xs">
                  <div className="border border-slate-200 dark:border-slate-800 rounded-md p-3 space-y-1">
                    <div className="font-semibold text-slate-900 dark:text-slate-50">Classification</div>
                    <div className="text-slate-500 dark:text-slate-400">Multi-class prediction (e.g., CIFAR-10, ImageNet)</div>
                    <div className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 mt-1">loss: "cross_entropy" · target_dtype: "long"</div>
                  </div>
                  <div className="border border-slate-200 dark:border-slate-800 rounded-md p-3 space-y-1">
                    <div className="font-semibold text-slate-900 dark:text-slate-50">Binary</div>
                    <div className="text-slate-500 dark:text-slate-400">Two-class prediction (e.g., spam/not-spam, cat/dog)</div>
                    <div className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 mt-1">loss: "bce_with_logits" · target_dtype: "float"</div>
                  </div>
                  <div className="border border-slate-200 dark:border-slate-800 rounded-md p-3 space-y-1">
                    <div className="font-semibold text-slate-900 dark:text-slate-50">Regression</div>
                    <div className="text-slate-500 dark:text-slate-400">Continuous value prediction (e.g., price, temperature)</div>
                    <div className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 mt-1">loss: "mse" | "mae" · target_dtype: "float"</div>
                  </div>
                </div>
              </div>

              {/* ---- Example Code ---- */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-50 mb-3">Complete Example — CIFAR-10 CNN (Classification)</h3>
              <div className="bg-slate-950 rounded-md border border-slate-800 overflow-hidden">
                <div className="flex items-center space-x-2 px-4 py-2 bg-slate-900 border-b border-slate-800">
                  <div className="w-3 h-3 rounded-full bg-rose-500" />
                  <div className="w-3 h-3 rounded-full bg-amber-500" />
                  <div className="w-3 h-3 rounded-full bg-emerald-500" />
                  <span className="text-xs font-mono text-slate-500 ml-2">model.py</span>
                </div>
                <pre className="p-4 text-xs font-mono text-slate-300 overflow-x-auto whitespace-pre">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# MESHML CONTRACT: Model Metadata
# The worker will parse this to dynamically configure the loss and metrics.
# =============================================================================
MODEL_METADATA = {
    # --- Required Base Fields ---
    "name": "cifar10-cnn",
    "version": "1.0",
    "framework": "pytorch",
    "input_shape": [3, 32, 32], # [Channels, Height, Width]
    "output_shape": [10],       # 10 classes
    
    # --- Task/Math Definition Fields ---
    "task_type": "classification",
    "loss": "cross_entropy",
    "metrics": ["accuracy"],
    "num_outputs": 10,  
    "target_dtype": "long" 
}

# =============================================================================
# PYTORCH ARCHITECTURE: CIFAR-10 CNN
# Input: 3 channels (RGB), 32x32 pixels.
# Output: 10 logits (one for each class).
# =============================================================================
class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        
        # Block 1: 3x32x32 -> 32x16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 32x16x16 -> 64x8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 64x8x8 -> 128x4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        # Flattened size: 128 channels * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Apply convolutions, ReLU, and pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Apply dense layers and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Return raw logits for CrossEntropyLoss
        
        return x

# =============================================================================
# MESHML CONTRACT: Entry Point
# The worker will call this function to instantiate the model.
# =============================================================================
def create_model():
    """
    Returns an instance of the model. MeshML workers will automatically 
    distribute this instance and wrap it in the training loop.
    """
    return CIFAR10Net()`}
                </pre>
              </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
