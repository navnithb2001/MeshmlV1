import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { authAPI } from '@/lib/api';

export default function Login() {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordMismatch, setPasswordMismatch] = useState(false);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(false);
    setPasswordMismatch(false);
    setLoading(true);

    try {
      if (!isLogin) {
        if (password !== confirmPassword) {
          setPasswordMismatch(true);
          setLoading(false);
          return;
        }

        // Register user first
        await authAPI.register({
          email,
          password,
          full_name: fullName || undefined,
        });
      }

      // Login to get access token (automatically after registration or directly for login)
      const response = await authAPI.login({
        email,
        password,
      });
      
      if (response && response.access_token) {
        localStorage.setItem('access_token', response.access_token);
        navigate('/workspace');
      }
    } catch (err) {
      // Force 1px red error border state
      setError(true);
      console.error(isLogin ? 'Login Failed' : 'Registration Failed', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950 p-4">
      {/* The Authentication Card */}
      <div
        className={clsx(
          "w-full max-w-sm bg-white dark:bg-slate-900 border p-8 shadow-sm transition-colors duration-200",
          error ? "border-rose-600 dark:border-rose-500" : "border-slate-200 dark:border-slate-800"
        )}
      >
        <div className="mb-8">
          <h1 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-slate-50 uppercase">
            MeshML Login
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {isLogin ? 'Login to your account' : 'Create a new account'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {!isLogin && (
            <div className="space-y-2">
              <label className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Full Name
              </label>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:border-cyan-600 dark:focus:border-cyan-400 transition-colors"
                placeholder="Full Name"
                required={!isLogin}
              />
            </div>
          )}

          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Email Address
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:border-cyan-600 dark:focus:border-cyan-400 transition-colors"
              placeholder="name@company.com"
              required
            />
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:border-cyan-600 dark:focus:border-cyan-400 transition-colors"
              placeholder="••••••••••••"
              required
              minLength={8}
            />
          </div>

          {!isLogin && (
            <div className="space-y-2">
              <label className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Confirm Password
              </label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-2 font-mono text-sm placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:border-cyan-600 dark:focus:border-cyan-400 transition-colors"
                placeholder="••••••••••••"
                required={!isLogin}
                minLength={8}
              />
            </div>
          )}

          {error && !passwordMismatch && (
            <div className="font-mono text-xs font-bold text-rose-600 dark:text-rose-500 mt-2">
              {isLogin ? 'ERR: INVALID_CREDENTIALS' : 'ERR: REGISTRATION_FAILED'}
            </div>
          )}
          {passwordMismatch && !isLogin && (
            <div className="font-mono text-xs font-bold text-rose-600 dark:text-rose-500 mt-2">
              ERR: PASSWORDS_DO_NOT_MATCH
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-slate-900 dark:bg-slate-50 text-slate-50 dark:text-slate-900 font-medium py-2 px-4 text-sm mt-4 hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {loading ? 'AUTHENTICATING...' : (isLogin ? 'LOG IN' : 'SIGN UP')}
          </button>
        </form>

        <div className="mt-8 text-center">
          <button
            type="button"
            onClick={() => {
              setIsLogin(!isLogin);
              setError(false);
              setPasswordMismatch(false);
            }}
            className="text-xs font-mono tracking-wider text-slate-500 hover:text-cyan-600 dark:hover:text-cyan-400 transition-colors"
          >
            {isLogin ? "DON'T HAVE AN ACCOUNT? SIGN UP" : 'ALREADY HAVE AN ACCOUNT? LOG IN'}
          </button>
        </div>
      </div>
    </div>
  );
}
