import { useQuery } from '@tanstack/react-query';
import { Download, HardDrive } from 'lucide-react';
import { modelsAPI } from '@/lib/api';
import { useToast } from '@/components/Toast';
import clsx from 'clsx';


interface Model {
  id: string;
  name: string;
  version: string;
  status: 'TRAINING' | 'COMPLETED' | 'FAILED';
  final_loss: number | null;
  size_bytes: number | null;
  created_at: string;
  task_type?: string | null; // from MODEL_METADATA, read-only
}

// Mocked fetcher function
const fetchModels = async () => {
  // Vault doesn't have a list endpoint exposed in API gateway yet, sticking to mock
  return [
    { id: 'mdl-abc123x', name: 'GPT-2 FineTuned', version: 'v1.4.2', status: 'COMPLETED', final_loss: 0.142, size_bytes: 483183820, created_at: '2023-10-26T14:30:00Z', task_type: 'classification' },
    { id: 'mdl-xyz987q', name: 'ResNet50 Hub', version: 'v2.0.0', status: 'COMPLETED', final_loss: 0.089, size_bytes: 102488310, created_at: '2023-10-25T09:15:00Z', task_type: 'regression' },
    { id: 'mdl-lmn456x', name: 'BERT Base Classification', version: 'v1.0.1', status: 'TRAINING', final_loss: null, size_bytes: null, created_at: '2023-10-27T10:00:00Z', task_type: 'binary' },
  ] as Model[];
};

export default function Vault() {
  const toast = useToast();
  const { data: models, isLoading } = useQuery({ queryKey: ['models'], queryFn: fetchModels });

  const formatBytes = (bytes: number | null) => {
    if (bytes === null) return '---';
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleDownload = async (modelId: string) => {
    console.log(`Triggering download for: ${modelId}`);
    try {
      // Use our typed client to get the signed URL
      await modelsAPI.getDownloadSignedUrl(modelId);
      toast.info(`Signed URL requested for model ${modelId}.`);
    } catch (err) {
      console.error(err);
      toast.error('Failed to request download URL.');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <HardDrive className="w-8 h-8 text-slate-900 dark:text-slate-50" />
        <div>
          <h1 className="text-2xl font-semibold tracking-tight uppercase text-slate-900 dark:text-slate-50">
            The Vault
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Model Registry & Artifacts
          </p>
        </div>
      </div>

      <div className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs uppercase bg-slate-50 dark:bg-slate-950 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
            <tr>
              <th className="px-6 py-4 font-mono font-medium">Model ID</th>
              <th className="px-6 py-4 font-medium">Name</th>
              <th className="px-6 py-4 font-medium">Version</th>
              <th className="px-6 py-4 font-medium">Final Loss</th>
              <th className="px-6 py-4 font-medium">Size</th>
              <th className="px-6 py-4 font-medium text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={6} className="px-6 py-8 text-center text-slate-500 font-mono text-xs">
                  FETCHING_REGISTRY...
                </td>
              </tr>
            ) : models?.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-8 text-center text-slate-500 font-mono text-xs">
                  NO_MODELS_FOUND
                </td>
              </tr>
            ) : (
              models?.map((model) => (
                <tr key={model.id} className="border-b border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-950 transition-colors">
                  <td className="px-6 py-4 font-mono text-slate-900 dark:text-slate-50">{model.id}</td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-700 dark:text-slate-300 font-medium text-sm">{model.name}</span>
                      {model.task_type && (
                        <span className={clsx(
                          "text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-sm",
                          model.task_type === 'classification' && "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-400",
                          model.task_type === 'regression' && "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-400",
                          model.task_type === 'binary' && "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-400",
                          !['classification','regression','binary'].includes(model.task_type) && "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
                        )}>
                          {model.task_type}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-slate-700 dark:text-slate-300 font-mono text-xs">{model.version}</td>
                  <td className="px-6 py-4 text-cyan-600 dark:text-cyan-400 font-mono text-xs">
                    {model.final_loss !== null ? model.final_loss.toFixed(4) : '---'}
                  </td>
                  <td className="px-6 py-4 text-slate-500 font-mono text-xs">
                    {formatBytes(model.size_bytes)}
                  </td>
                  <td className="px-6 py-4 text-right">
                    {model.status === 'COMPLETED' ? (
                      <button
                        onClick={() => handleDownload(model.id)}
                        className="inline-flex items-center space-x-1 text-xs font-medium bg-slate-900 dark:bg-slate-50 text-slate-50 dark:text-slate-900 px-3 py-1 hover:opacity-90 transition-opacity uppercase"
                      >
                        <Download className="w-3 h-3" />
                        <span>Download (.pt)</span>
                      </button>
                    ) : (
                       <span className="font-mono text-xs text-slate-400 dark:text-slate-500 uppercase">
                         {model.status}
                       </span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
