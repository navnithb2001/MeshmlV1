import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Key, X } from 'lucide-react';
import { authAPI } from '@/lib/api';
import { useToast } from '@/components/Toast';

export default function ChangePasswordModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const toast = useToast();

  const changePassword = useMutation({
    mutationFn: () => authAPI.changePassword({ old_password: oldPassword, new_password: newPassword }),
    onSuccess: () => {
      toast.success('Password changed successfully.');
      onClose();
      // Reset form
      setOldPassword('');
      setNewPassword('');
      setConfirmPassword('');
    },
    onError: (err: any) => {
      toast.error(err?.response?.data?.detail || 'Failed to change password.');
    }
  });

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword.length < 8) {
      toast.warning('New password must be at least 8 characters long.');
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.warning('New passwords do not match.');
      return;
    }
    changePassword.mutate();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-md bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xl flex flex-col animate-in zoom-in-95 duration-150 rounded-sm">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 p-5 shrink-0">
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50 flex items-center">
            <Key className="w-5 h-5 mr-2 text-cyan-500" />
            Change Password
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400 block">Current Password</label>
            <input 
              type="password" 
              required
              value={oldPassword}
              onChange={(e) => setOldPassword(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm focus:outline-none focus:border-cyan-500 transition-colors rounded-sm"
              placeholder="••••••••"
            />
          </div>
          
          <div className="space-y-1.5">
            <label className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400 block">New Password</label>
            <input 
              type="password" 
              required
              minLength={8}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm focus:outline-none focus:border-cyan-500 transition-colors rounded-sm"
              placeholder="••••••••"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400 block">Confirm New Password</label>
            <input 
              type="password" 
              required
              minLength={8}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm focus:outline-none focus:border-cyan-500 transition-colors rounded-sm"
              placeholder="••••••••"
            />
          </div>

          <div className="pt-4 flex justify-end gap-3 border-t border-slate-100 dark:border-slate-800 mt-6 pt-5">
            <button 
              type="button" 
              onClick={onClose} 
              className="px-4 py-2 text-sm font-medium border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 rounded-sm transition-colors"
            >
              Cancel
            </button>
            <button 
              type="submit" 
              disabled={changePassword.isPending}
              className="px-4 py-2 text-sm font-medium bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 text-white rounded-sm transition-colors flex items-center justify-center min-w-[120px]"
            >
              {changePassword.isPending ? 'Updating...' : 'Save Password'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
