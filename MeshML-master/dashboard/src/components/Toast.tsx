import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from 'react';
import { X, CheckCircle, AlertTriangle, AlertCircle, Info } from 'lucide-react';
import clsx from 'clsx';

// ==========================================
// Types
// ==========================================

type ToastType = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration: number;
}

interface ToastContextValue {
  success: (message: string, duration?: number) => void;
  error: (message: string, duration?: number) => void;
  warning: (message: string, duration?: number) => void;
  info: (message: string, duration?: number) => void;
}

// ==========================================
// Context
// ==========================================

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within a ToastProvider');
  return ctx;
}

// Global emitter for non-React code (API interceptor)
type Listener = (message: string, type: ToastType) => void;
const listeners: Set<Listener> = new Set();

export const toastEmitter = {
  emit: (message: string, type: ToastType) => {
    listeners.forEach((fn) => fn(message, type));
  },
  subscribe: (fn: Listener) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

// ==========================================
// Individual Toast Item
// ==========================================

const icons: Record<ToastType, typeof CheckCircle> = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

const colors: Record<ToastType, string> = {
  success: 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-200',
  error: 'border-rose-500 bg-rose-50 dark:bg-rose-950/60 text-rose-800 dark:text-rose-200',
  warning: 'border-amber-500 bg-amber-50 dark:bg-amber-950/60 text-amber-800 dark:text-amber-200',
  info: 'border-cyan-500 bg-cyan-50 dark:bg-cyan-950/60 text-cyan-800 dark:text-cyan-200',
};

const iconColors: Record<ToastType, string> = {
  success: 'text-emerald-500',
  error: 'text-rose-500',
  warning: 'text-amber-500',
  info: 'text-cyan-500',
};

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  const Icon = icons[toast.type];

  useEffect(() => {
    const timer = setTimeout(() => onDismiss(toast.id), toast.duration);
    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, onDismiss]);

  return (
    <div
      className={clsx(
        'flex items-start gap-3 px-4 py-3 border-l-4 shadow-lg backdrop-blur-sm',
        'animate-in slide-in-from-right duration-300',
        'max-w-md w-full',
        colors[toast.type]
      )}
    >
      <Icon className={clsx('w-5 h-5 shrink-0 mt-0.5', iconColors[toast.type])} />
      <p className="text-sm font-medium flex-1">{toast.message}</p>
      <button
        onClick={() => onDismiss(toast.id)}
        className="shrink-0 opacity-60 hover:opacity-100 transition-opacity"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

// ==========================================
// Provider
// ==========================================

let idCounter = 0;

function normalizeToastMessage(message: unknown): string {
  if (typeof message === 'string') return message;
  if (typeof message === 'number' || typeof message === 'boolean') return String(message);
  if (message == null) return 'Unexpected error.';

  if (Array.isArray(message)) {
    const parts = message
      .map((item) => normalizeToastMessage(item))
      .filter((item) => item && item !== 'Unexpected error.');
    return parts.length ? parts.join(' | ') : 'Unexpected error.';
  }

  if (typeof message === 'object') {
    const candidate = message as Record<string, unknown>;
    if (typeof candidate.message === 'string') return candidate.message;
    if (typeof candidate.msg === 'string') return candidate.msg;
    if (candidate.detail !== undefined) return normalizeToastMessage(candidate.detail);
    try {
      return JSON.stringify(message);
    } catch {
      return 'Unexpected error.';
    }
  }

  return 'Unexpected error.';
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback((message: unknown, type: ToastType, duration = 5000) => {
    const id = `toast-${++idCounter}`;
    setToasts((prev) => [...prev, { id, message: normalizeToastMessage(message), type, duration }]);
  }, []);

  const api: ToastContextValue = {
    success: useCallback((msg, dur) => addToast(msg, 'success', dur), [addToast]),
    error: useCallback((msg, dur) => addToast(msg, 'error', dur ?? 7000), [addToast]),
    warning: useCallback((msg, dur) => addToast(msg, 'warning', dur), [addToast]),
    info: useCallback((msg, dur) => addToast(msg, 'info', dur), [addToast]),
  };

  // Subscribe to emitter for non-React code
  useEffect(() => {
    const unsub = toastEmitter.subscribe((message, type) => addToast(message, type));
    return () => { unsub(); };
  }, [addToast]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      {/* Toast container — fixed top-right */}
      <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-auto">
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}
