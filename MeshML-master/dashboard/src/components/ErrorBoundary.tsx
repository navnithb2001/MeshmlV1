import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary]', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950 p-6">
          <div className="max-w-lg w-full border border-rose-300 dark:border-rose-800 bg-white dark:bg-slate-900 p-8 text-center space-y-6">
            <div className="flex justify-center">
              <AlertTriangle className="w-12 h-12 text-rose-500" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-900 dark:text-slate-50 uppercase tracking-wider">
                Something Went Wrong
              </h2>
              <p className="text-sm text-slate-500 mt-2">
                An unexpected error occurred. This has been logged for debugging.
              </p>
            </div>
            {this.state.error && (
              <div className="bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-4 text-left">
                <p className="text-xs font-mono text-rose-600 dark:text-rose-400 break-all">
                  {this.state.error.message}
                </p>
              </div>
            )}
            <button
              onClick={this.handleRetry}
              className="inline-flex items-center gap-2 bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-900 text-sm font-semibold py-3 px-6 uppercase tracking-wider transition-colors hover:opacity-90"
            >
              <RotateCcw className="w-4 h-4" />
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
