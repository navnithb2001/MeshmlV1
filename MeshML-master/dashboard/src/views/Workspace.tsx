import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FolderGit2, Plus, KeySquare, X, ArrowRight } from 'lucide-react';
import { groupsAPI } from '@/lib/api';

export default function Workspace() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [modalMode, setModalMode] = useState<'create' | 'join' | null>(null);

  // Form states
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [workerId, setWorkerId] = useState('');
  const [invitationCode, setInvitationCode] = useState('');

  const { data: groups, isLoading, isError, error } = useQuery({
    queryKey: ['groups'],
    queryFn: groupsAPI.listGroups,
  });

  const createGroup = useMutation({
    mutationFn: groupsAPI.createGroup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['groups'] });
      setModalMode(null);
      setName('');
      setDescription('');
    },
  });

  const joinGroup = useMutation({
    mutationFn: groupsAPI.acceptInvitation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['groups'] });
      setModalMode(null);
      setWorkerId('');
      setInvitationCode('');
    },
  });

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50 tracking-tight">Your Workspace</h1>
          <p className="text-sm font-mono text-slate-500 mt-1">Select an active group or create a new one.</p>
        </div>
        <div className="flex space-x-3">
          <button 
            onClick={() => setModalMode('join')}
            className="border border-slate-200 dark:border-slate-800 text-slate-900 dark:text-slate-50 text-sm font-medium py-2 px-4 hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors flex items-center space-x-2"
          >
            <KeySquare className="w-4 h-4" />
            <span>JOIN</span>
          </button>
          <button 
            onClick={() => setModalMode('create')}
            className="bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-900 text-sm font-medium py-2 px-4 hover:bg-slate-800 dark:hover:bg-slate-200 transition-colors flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>NEW GROUP</span>
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center space-x-2 text-slate-500 font-mono text-sm py-12">
          <div className="w-4 h-4 border-2 border-slate-200 border-t-slate-900 dark:border-slate-700 dark:border-t-slate-50 rounded-full animate-spin" />
          <span>Fetching groups...</span>
        </div>
      ) : isError ? (
        <div className="col-span-full border border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/20 p-8 flex flex-col items-start justify-center text-rose-700 dark:text-rose-400">
          <h3 className="font-bold flex items-center mb-2"><FolderGit2 className="w-5 h-5 mr-2" /> Connection Refused</h3>
          <p className="text-sm font-mono opacity-80 mb-4">The server returned an error (likely 401/403) when attempting to fetch the workspace.</p>
          <p className="text-xs font-mono bg-rose-100 dark:bg-rose-950 p-2 border border-rose-200 dark:border-rose-800 rounded w-full overflow-x-auto">
            {String(error)}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {(groups || []).map((group) => (
            <div 
              key={group.id} 
              onClick={() => navigate(`/groups/${group.id}`)}
              className="group border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-6 flex flex-col cursor-pointer hover:border-cyan-500 dark:hover:border-cyan-500 transition-colors relative"
            >
              <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity translate-x-1 group-hover:translate-x-0 text-cyan-500">
                <ArrowRight className="w-5 h-5" />
              </div>
              <div className="bg-slate-100 dark:bg-slate-900 w-12 h-12 flex flex-col items-center justify-center mb-4 text-slate-400 group-hover:text-cyan-500 transition-colors">
                <FolderGit2 className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-slate-50 tracking-tight">{group.name}</h3>
              <p className="text-sm font-mono text-slate-500 line-clamp-2 mt-2 flex-grow">
                {group.description || 'No description provided.'}
              </p>
              <div className="mt-6 pt-4 border-t border-slate-100 dark:border-slate-800 flex justify-between items-center text-xs font-mono text-slate-400">
                <span>{group.is_public ? 'PUBLIC' : 'PRIVATE'}</span>
                <span>{new Date(group.created_at).toISOString().split('T')[0]}</span>
              </div>
            </div>
          ))}
          
          {(!groups || groups.length === 0) && (
            <div className="col-span-full border border-dashed border-slate-300 dark:border-slate-800 p-12 flex flex-col items-center justify-center bg-slate-50/50 dark:bg-slate-950/50">
              <FolderGit2 className="w-8 h-8 text-slate-300 dark:text-slate-700 mb-4" />
              <p className="text-sm font-mono text-slate-500">No active groups found in this workspace.</p>
            </div>
          )}
        </div>
      )}

      {/* Modals */}
      {modalMode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="w-full max-w-md bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 flex flex-col shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 p-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900 dark:text-slate-50">
                {modalMode === 'create' ? 'Create New Group' : 'Join Existing Group'}
              </h2>
              <button onClick={() => setModalMode(null)} className="p-1 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors text-slate-500">
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div className="p-6 space-y-4">
              {modalMode === 'create' ? (
                <>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-slate-500">Group Name</label>
                    <input 
                      type="text" 
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm focus:outline-none focus:border-cyan-600 transition-colors"
                      placeholder="e.g. cv-research-team"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-slate-500">Description</label>
                    <textarea 
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      rows={3}
                      className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm focus:outline-none focus:border-cyan-600 transition-colors resize-none"
                      placeholder="Optional details about this group..."
                    />
                  </div>
                </>
              ) : (
                <>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-slate-500">Invitation Code</label>
                    <input 
                      type="text" 
                      value={invitationCode}
                      onChange={(e) => setInvitationCode(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm focus:outline-none focus:border-cyan-600 transition-colors"
                      placeholder="Enter the 16-character code"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wider text-slate-500">Bind Worker ID</label>
                    <input 
                      type="text" 
                      value={workerId}
                      onChange={(e) => setWorkerId(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm focus:outline-none focus:border-cyan-600 transition-colors"
                      placeholder="e.g. worker-node-01"
                    />
                  </div>
                </>
              )}
            </div>

            <div className="border-t border-slate-200 dark:border-slate-800 p-4 bg-slate-50 dark:bg-slate-900/50 flex justify-end space-x-3">
              <button onClick={() => setModalMode(null)} className="text-sm font-medium text-slate-500 hover:text-slate-900 dark:hover:text-slate-50 transition-colors px-4 py-2">
                CANCEL
              </button>
              <button 
                onClick={() => modalMode === 'create' ? createGroup.mutate({ name, description }) : joinGroup.mutate({ worker_id: workerId, invitation_code: invitationCode })}
                disabled={createGroup.isPending || joinGroup.isPending || (modalMode === 'create' && !name.trim()) || (modalMode === 'join' && (!invitationCode.trim() || !workerId.trim()))}
                className="bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-900 text-sm font-medium py-2 px-6 disabled:opacity-50 transition-colors"
              >
                {createGroup.isPending || joinGroup.isPending ? 'PROCESSING...' : 'CONFIRM'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
