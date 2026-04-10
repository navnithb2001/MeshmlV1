import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Play, Activity, Settings as SettingsIcon, Users, Key, Save, UserCog, Trash2, Database } from 'lucide-react';
import clsx from 'clsx';
import { jobsAPI, workersAPI, groupsAPI, authAPI, datasetsAPI } from '@/lib/api';
import SetupModal from '@/components/SetupModal';
import ConfirmModal from '@/components/ConfirmModal';
import { useToast } from '@/components/Toast';

export default function GroupDashboard() {
  const { groupId } = useParams<{ groupId: string }>();
  const navigate = useNavigate();
  const toast = useToast();
  const [activeTab, setActiveTab] = useState<'jobs' | 'datasets' | 'workers' | 'settings'>('jobs');
  const [isSetupOpen, setIsSetupOpen] = useState(false);
  const [copiedContext, setCopiedContext] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [confirmState, setConfirmState] = useState<
    | { type: 'deleteDataset'; datasetId: string; datasetName: string }
    | { type: 'deleteGroup' }
    | null
  >(null);
  const queryClient = useQueryClient();

  // Queries
  const { data: currentUser } = useQuery({
    queryKey: ['me'],
    queryFn: () => authAPI.getCurrentUser(),
  });

  const { data: group } = useQuery({
    queryKey: ['group', groupId],
    queryFn: () => groupsAPI.getGroup(groupId!),
    enabled: !!groupId,
  });

  const { data: members, isLoading: isMembersLoading } = useQuery({
    queryKey: ['groupMembers', groupId],
    queryFn: () => groupsAPI.getGroupMembers(groupId!),
    enabled: !!groupId && activeTab === 'settings',
  });

  useEffect(() => {
    if (group?.name && !editName) {
      setEditName(group.name);
    }
  }, [group]);
  const { data: jobs, isLoading: isJobsLoading } = useQuery({
    queryKey: ['jobs', groupId],
    queryFn: () => jobsAPI.listJobs({ group_id: groupId }),
    enabled: !!groupId && activeTab === 'jobs',
    refetchInterval: 5000,
  });

  const { data: workers, isLoading: isWorkersLoading } = useQuery({
    queryKey: ['workers', groupId],
    queryFn: () => workersAPI.listWorkers({ group_id: groupId }),
    enabled: !!groupId && activeTab === 'workers',
    refetchInterval: 10000,
  });

  const { data: datasets, isLoading: isDatasetsLoading } = useQuery({
    queryKey: ['datasets', groupId],
    queryFn: () => datasetsAPI.listDatasets(),
    enabled: !!groupId && activeTab === 'datasets',
    refetchInterval: 10000,
  });

  const generateInvite = useMutation({
    mutationFn: (data: { max_uses?: number; expires_in_hours?: number }) => 
      groupsAPI.createInvitation(groupId!, data),
    onSuccess: (data) => {
      setCopiedContext(`Code: ${data.code}`);
      navigator.clipboard.writeText(data.code);
      setTimeout(() => setCopiedContext(null), 3000);
    }
  });

  const updateGroup = useMutation({
    mutationFn: (data: { name: string }) => groupsAPI.updateGroup(groupId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['group', groupId] });
      queryClient.invalidateQueries({ queryKey: ['groups'] });
    }
  });

  const deleteDataset = useMutation({
    mutationFn: (datasetId: string) => datasetsAPI.deleteDataset(datasetId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets', groupId] });
      toast.success('Dataset deleted.');
    },
    onError: () => {
      toast.error('Failed to delete dataset.');
    },
  });

  if (!groupId) return null;

  const currentMember = members?.find(m => m.user_id === currentUser?.id);
  const isOwner = group?.owner_id === currentUser?.id || currentMember?.role === 'owner';
  const isAdminOrOwner = isOwner || currentMember?.role === 'admin';

  return (
    <div className="flex flex-col h-full animate-in fade-in duration-300">
      {/* Header & Tabs */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4 mb-6">
        <div className="flex items-center justify-between mb-4 mt-2">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50 tracking-tight flex items-center space-x-2">
              <span>{group?.name || 'Group Dashboard'}</span>
            </h1>
          </div>
          {activeTab === 'jobs' && isOwner && (
            <button 
              onClick={() => setIsSetupOpen(true)}
              className="bg-cyan-600 hover:bg-cyan-700 text-white text-sm font-medium py-2 px-4 transition-colors flex items-center space-x-2 shadow-sm"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>NEW TRAINING RUN</span>
            </button>
          )}
        </div>
        
        <div className="flex space-x-6">
          <button 
            onClick={() => setActiveTab('jobs')}
            className={clsx(
              "flex items-center space-x-2 pb-2 text-sm font-medium uppercase tracking-wider transition-colors",
              activeTab === 'jobs' 
                ? "text-cyan-600 dark:text-cyan-400 border-b-2 border-cyan-600 dark:border-cyan-400" 
                : "text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 border-b-2 border-transparent"
            )}
          >
            <Activity className="w-4 h-4" />
            <span>Jobs</span>
          </button>
          <button 
            onClick={() => setActiveTab('datasets')}
            className={clsx(
              "flex items-center space-x-2 pb-2 text-sm font-medium uppercase tracking-wider transition-colors",
              activeTab === 'datasets' 
                ? "text-cyan-600 dark:text-cyan-400 border-b-2 border-cyan-600 dark:border-cyan-400" 
                : "text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 border-b-2 border-transparent"
            )}
          >
            <Database className="w-4 h-4" />
            <span>Datasets</span>
          </button>
          <button 
            onClick={() => setActiveTab('workers')}
            className={clsx(
              "flex items-center space-x-2 pb-2 text-sm font-medium uppercase tracking-wider transition-colors",
              activeTab === 'workers' 
                ? "text-cyan-600 dark:text-cyan-400 border-b-2 border-cyan-600 dark:border-cyan-400" 
                : "text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 border-b-2 border-transparent"
            )}
          >
            <Users className="w-4 h-4" />
            <span>Workers</span>
          </button>
          {isOwner && (
            <button 
              onClick={() => setActiveTab('settings')}
              className={clsx(
                "flex items-center space-x-2 pb-2 text-sm font-medium uppercase tracking-wider transition-colors",
                activeTab === 'settings' 
                  ? "text-cyan-600 dark:text-cyan-400 border-b-2 border-cyan-600 dark:border-cyan-400" 
                  : "text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 border-b-2 border-transparent"
              )}
            >
              <SettingsIcon className="w-4 h-4" />
              <span>Settings</span>
            </button>
          )}
        </div>
      </div>

      {/* Tab Panels */}
      <div className="flex-1 overflow-auto">
        
        {/* Jobs Tab */}
        {activeTab === 'jobs' && (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 dark:bg-slate-950/50 border-b border-slate-200 dark:border-slate-800 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  <th className="px-6 py-4 font-mono">Job ID</th>
                  <th className="px-6 py-4">Dataset ID</th>
                  <th className="px-6 py-4">Model Config</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4">Created Date</th>
                  <th className="px-6 py-4">Action</th>
                </tr>
              </thead>
              <tbody>
                {isJobsLoading && (
                  <tr>
                    <td colSpan={6} className="px-6 py-8 text-center text-sm font-mono text-slate-400">Loading jobs...</td>
                  </tr>
                )}
                {(!isJobsLoading && (!jobs || jobs.length === 0)) && (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-sm font-mono text-slate-500">
                      No jobs running in this group.<br/>Click "New Training Run" to start.
                    </td>
                  </tr>
                )}
                {(jobs || []).map((job) => (
                  <tr key={job.id} className="border-b border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-950 transition-colors">
                    <td className="px-6 py-4 font-mono text-slate-900 dark:text-slate-50 text-sm">{job.id}</td>
                    <td className="px-6 py-4 text-slate-700 dark:text-slate-300 font-mono text-xs">{job.dataset_id || '---'}</td>
                    <td className="px-6 py-4 text-slate-700 dark:text-slate-300 font-mono text-xs">{job.model_id || '---'}</td>
                    <td className="px-6 py-4">
                      <span 
                        title={['FAILED', 'CANCELLED'].includes((job.status || '').toUpperCase()) ? (job as any).error_message || 'Job failed' : undefined}
                        className={clsx(
                        "font-mono text-xs font-bold uppercase tracking-wider",
                        (() => {
                          const s = (job.status || '').toUpperCase();
                          if (s === 'COMPLETED') return "text-emerald-500";
                          if (s === 'ACTIVE' || s === 'TRAINING') return "text-cyan-500";
                          if (s === 'WAITING' || s === 'PENDING') return "text-amber-500";
                          if (s === 'FAILED' || s === 'CANCELLED') return "text-rose-500";
                          return "text-slate-500";
                        })()
                      )}>
                        {job.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-500 font-mono text-xs text-slate-400">
                      {new Date(job.created_at).toISOString().split('T')[0]}
                    </td>
                    <td className="px-6 py-4">
                      <Link 
                        to={`/jobs/${job.id}/live`}
                        className="text-cyan-600 hover:text-cyan-800 dark:hover:text-cyan-400 font-medium text-xs uppercase tracking-wider"
                      >
                        View Job
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Datasets Tab */}
        {activeTab === 'datasets' && (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 dark:bg-slate-950/50 border-b border-slate-200 dark:border-slate-800 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  <th className="px-6 py-4 font-mono">Dataset ID</th>
                  <th className="px-6 py-4">Name</th>
                  <th className="px-6 py-4">Format</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4">Created Date</th>
                  <th className="px-6 py-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {isDatasetsLoading && (
                  <tr>
                    <td colSpan={5} className="px-6 py-8 text-center text-sm font-mono text-slate-400">Loading datasets...</td>
                  </tr>
                )}
                  {(!isDatasetsLoading && (!datasets?.datasets || datasets.datasets.length === 0)) && (
                    <tr>
                      <td colSpan={6} className="px-6 py-12 text-center text-sm font-mono text-slate-500">
                        No datasets uploaded yet.
                      </td>
                    </tr>
                  )}
                  {(datasets?.datasets || []).map((dataset) => (
                    <tr key={dataset.id} className="border-b border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-950 transition-colors">
                      <td className="px-6 py-4 font-mono text-slate-900 dark:text-slate-50 text-sm">{dataset.id}</td>
                      <td className="px-6 py-4 text-slate-700 dark:text-slate-300 text-sm">{dataset.name}</td>
                      <td className="px-6 py-4 text-slate-700 dark:text-slate-300 font-mono text-xs">{dataset.format || '---'}</td>
                      <td className="px-6 py-4">
                        <span className={clsx(
                          "font-mono text-xs font-bold uppercase tracking-wider",
                          (() => {
                            const s = (dataset.status || '').toUpperCase();
                            if (s === 'AVAILABLE' || s === 'UPLOADED') return "text-emerald-500";
                            if (s === 'PENDING' || s === 'SHARDING') return "text-amber-500";
                            if (s === 'FAILED') return "text-rose-500";
                            return "text-slate-500";
                          })()
                        )}>
                          {dataset.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-slate-500 font-mono text-xs">
                        {new Date(dataset.created_at).toISOString().split('T')[0]}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={() => {
                            setConfirmState({ type: 'deleteDataset', datasetId: dataset.id, datasetName: dataset.name });
                          }}
                          disabled={deleteDataset.isPending}
                          className="inline-flex items-center space-x-1 text-xs font-medium text-rose-500 hover:text-rose-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                          title="Delete dataset"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                          <span>Delete</span>
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Workers Tab */}
        {activeTab === 'workers' && (
          <div className="space-y-6">


            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-950/50 border-b border-slate-200 dark:border-slate-800 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    <th className="px-6 py-4 font-mono">Worker Node ID</th>
                    <th className="px-6 py-4">Status</th>
                    <th className="px-6 py-4">Heartbeat</th>
                    <th className="px-6 py-4">Registered</th>
                  </tr>
                </thead>
                <tbody>
                  {isWorkersLoading && (
                    <tr>
                      <td colSpan={4} className="px-6 py-8 text-center text-sm font-mono text-slate-400">Loading active nodes...</td>
                    </tr>
                  )}
                  {(!isWorkersLoading && (!workers || workers.length === 0)) && (
                    <tr>
                      <td colSpan={4} className="px-6 py-12 text-center text-sm font-mono text-slate-500">
                        No active compute nodes found. Connect a worker via CLI to start training.
                      </td>
                    </tr>
                  )}
                  {(workers || []).map((worker) => (
                    <tr key={worker.id} className="border-b border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-950 transition-colors">
                      <td className="px-6 py-4 font-mono text-slate-900 dark:text-slate-50 text-sm">{worker.worker_id}</td>
                      <td className="px-6 py-4">
                        <span className="flex items-center space-x-2">
                          <span className={clsx(
                            "w-2 h-2 rounded-full",
                            worker.status === 'idle' ? "bg-amber-400" :
                            worker.status === 'training' ? "bg-cyan-500 animate-pulse" :
                            "bg-slate-500"
                          )}></span>
                          <span className="font-mono text-xs uppercase text-slate-600 dark:text-slate-300 tracking-wider bg-slate-100 dark:bg-slate-800 px-2 py-1">{worker.status}</span>
                        </span>
                      </td>
                      <td className="px-6 py-4 text-slate-700 dark:text-slate-300 text-xs font-mono">
                         {worker.last_heartbeat ? new Date(worker.last_heartbeat).toLocaleTimeString() : '---'}
                      </td>
                      <td className="px-6 py-4 text-slate-500 font-mono text-xs">
                        {new Date(worker.created_at).toISOString().split('T')[0]}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div className="max-w-4xl space-y-6">

            {/* Profile / Details */}
            {isAdminOrOwner && (
              <div className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-8">
                <h3 className="text-lg font-bold text-slate-900 dark:text-slate-50 flex items-center mb-6">
                  <SettingsIcon className="w-5 h-5 mr-2 text-cyan-500" /> Group Details
                </h3>
                
                <div className="space-y-4 max-w-xl">
                  <div>
                    <label className="block text-xs font-mono text-slate-500 mb-1 uppercase tracking-wider">Group Name</label>
                    <div className="flex space-x-2">
                      <input 
                        type="text" 
                        value={editName}
                        onChange={e => setEditName(e.target.value)}
                        className="flex-1 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-mono focus:outline-none focus:border-cyan-500 transition-colors"
                      />
                      <button 
                        onClick={() => updateGroup.mutate({ name: editName })}
                        disabled={updateGroup.isPending || editName === group?.name}
                        className="bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 text-white px-4 flex items-center justify-center transition-colors"
                      >
                        <Save className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Members Management */}
            {isAdminOrOwner && (
              <div className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-8">
                <h3 className="text-lg font-bold text-slate-900 dark:text-slate-50 flex items-center mb-6">
                  <UserCog className="w-5 h-5 mr-2 text-cyan-500" /> Member Info
                </h3>
                
                <div className="border border-slate-200 dark:border-slate-800">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="bg-slate-50 dark:bg-slate-950/50 border-b border-slate-200 dark:border-slate-800 uppercase text-xs font-semibold tracking-wider text-slate-500">
                        <th className="px-4 py-3">Member Email / Worker Name</th>
                        <th className="px-4 py-3">Role</th>
                      </tr>
                    </thead>
                    <tbody>
                      {isMembersLoading && (
                        <tr><td colSpan={3} className="px-4 py-6 text-center text-slate-400 font-mono text-xs">Loading members...</td></tr>
                      )}
                      {(members || []).map(member => (
                        <tr key={member.id} className="border-b border-slate-200 dark:border-slate-800 last:border-0 hover:bg-slate-50 dark:hover:bg-slate-950">
                          <td className="px-4 py-3 font-mono text-xs">
                            {member.user_id ? `${member.user?.email || 'Unknown User'}` : `Worker: ${member.worker_id}`}
                          </td>
                          <td className="px-4 py-3">
                            <span className={clsx(
                              "text-xs font-mono px-2 py-1 uppercase tracking-wider",
                              member.role === 'owner' ? "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400" :
                              member.role === 'admin' ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400" :
                              "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                            )}>
                              {member.role}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Access control box */}
            <div className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-8">
              <h3 className="text-lg font-bold text-slate-900 dark:text-slate-50 flex items-center mb-2">
                <Key className="w-5 h-5 mr-2 text-cyan-500" /> Group Access Invitation
              </h3>
              <p className="text-sm font-mono text-slate-500 mb-6">
                Generate a new 16-character invitation code to allow other engineers or worker nodes to securely join this workspace.
              </p>
              
              <div className="pt-4 border-t border-slate-100 dark:border-slate-800 max-w-xl">
                <button 
                  disabled={generateInvite.isPending || !isAdminOrOwner}
                  onClick={() => generateInvite.mutate({ expires_in_hours: 168 })}
                  className="bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-900 text-sm font-medium py-3 px-6 w-full disabled:opacity-50 transition-colors"
                >
                  {generateInvite.isPending ? 'GENERATING...' : 'GENERATE 7-DAY INVITE CODE'}
                </button>
              </div>
              
              {copiedContext?.startsWith('Code:') && (
                <div className="mt-4 max-w-xl bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800/50 p-4 flex items-center justify-between text-sm">
                  <span className="font-mono text-emerald-800 dark:text-emerald-400">{copiedContext.replace('Code: ', '')}</span>
                  <span className="text-xs uppercase tracking-wider font-bold text-emerald-600 dark:text-emerald-500">Copied to clipboard!</span>
                </div>
              )}
            </div>

            {/* Danger Zone */}
            {isOwner && (
              <div className="border border-rose-300 dark:border-rose-800 bg-white dark:bg-slate-900 p-8">
                <h3 className="text-lg font-bold text-rose-600 dark:text-rose-400 flex items-center mb-2">
                  <Trash2 className="w-5 h-5 mr-2" /> Danger Zone
                </h3>
                <p className="text-sm text-slate-500 mb-6">
                  Permanently delete this group and all associated data. This action cannot be undone.
                </p>
                <button
                  onClick={() => setConfirmState({ type: 'deleteGroup' })}
                  className="bg-rose-600 hover:bg-rose-700 text-white text-sm font-semibold py-3 px-6 uppercase tracking-wider transition-colors"
                >
                  Delete Group
                </button>
              </div>
            )}

          </div>
        )}

      </div>
      
      {/* Contextual Setup Modal */}
      <SetupModal isOpen={isSetupOpen} onClose={() => setIsSetupOpen(false)} />

      {/* Unified Confirm Modal */}
      {(() => {
        let activeDatasetJobs = 0;
        if (confirmState?.type === 'deleteDataset' && jobs) {
          activeDatasetJobs = jobs.filter((j) => 
            j.dataset_id === confirmState.datasetId && 
            ['pending', 'processing', 'running', 'sharding', 'uploading'].includes((j.status || '').toLowerCase())
          ).length;
        }

        return (
          <ConfirmModal
            isOpen={confirmState !== null}
            title={
              confirmState?.type === 'deleteDataset'
                ? activeDatasetJobs > 0 ? 'DESTRUCTIVE_ACTION_DETECTED' : 'Delete Dataset'
                : 'Delete Group'
            }
            message={
              confirmState?.type === 'deleteDataset'
                ? activeDatasetJobs > 0 
                  ? `This dataset is currently being used by ${activeDatasetJobs} active training jobs. Deleting it will permanently cancel these runs. Are you sure? (Completed or Failed jobs will remain in your history).`
                  : 'Are you sure you want to permanently delete this dataset? This action cannot be undone.'
                : 'Are you sure you want to permanently delete this group and all associated data? This action cannot be undone.'
            }
            detail={
              confirmState?.type === 'deleteDataset'
                ? confirmState.datasetName
                : group?.name
            }
            confirmLabel={confirmState?.type === 'deleteDataset' ? activeDatasetJobs > 0 ? 'Force Delete' : 'Delete Dataset' : 'Delete Group'}
        onCancel={() => setConfirmState(null)}
        onConfirm={async () => {
          if (!confirmState) return;
          if (confirmState.type === 'deleteDataset') {
            deleteDataset.mutate(confirmState.datasetId);
            setConfirmState(null);
          } else if (confirmState.type === 'deleteGroup') {
            setConfirmState(null);
            try {
              await groupsAPI.deleteGroup(groupId!);
              navigate('/workspace');
            } catch (err: any) {
              toast.error(err?.response?.data?.detail || 'Failed to delete group.');
            }
          }
        }}
        />
        );
      })()}
    </div>
  );
}
