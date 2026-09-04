import React, { useState, useEffect } from 'react';
import { Activity, Database, Server, RefreshCw, XCircle, Shield } from 'lucide-react';
import axios from 'axios';

export default function HealthStatus() {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(null);

  const fetchHealth = async () => {
    setLoading(true);
    setError(null);
    const startTime = Date.now();
    try {
      // Try Vite proxy /api/v1/health first, fallback to direct localhost:8000
      let response;
      try {
        response = await axios.get('/api/v1/health', { timeout: 3000 });
      } catch (err) {
        response = await axios.get('http://127.0.0.1:8000/api/v1/health', { timeout: 3000 });
      }
      const endTime = Date.now();
      setLatency(endTime - startTime);
      setHealthData(response.data);
    } catch (err) {
      console.error('Failed to fetch backend health:', err);
      setError(err.message || 'Cannot reach FastAPI backend server');
      setHealthData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full max-w-4xl mx-auto p-6 rounded-2xl glass-card border border-slate-800 shadow-2xl">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-600/20 text-blue-400 rounded-xl border border-blue-500/30">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white tracking-wide">RazorGuard AI System Status</h2>
            <p className="text-xs text-slate-400">Phase 1 Monorepo Verification & API Connection</p>
          </div>
        </div>
        <button
          onClick={fetchHealth}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-xl transition border border-slate-700 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh Status
        </button>
      </div>

      {error ? (
        <div className="p-4 bg-red-950/40 border border-red-800/50 rounded-xl flex items-center justify-between">
          <div className="flex items-center gap-3">
            <XCircle className="w-6 h-6 text-red-500 shrink-0" />
            <div>
              <h3 className="font-semibold text-red-400">Backend Connection Failed</h3>
              <p className="text-xs text-red-300/80">{error}</p>
            </div>
          </div>
          <span className="text-xs px-3 py-1 bg-red-900/60 text-red-300 rounded-full font-mono">
            DISCONNECTED
          </span>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Status Badge */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-emerald-500/10 text-emerald-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs text-slate-400 font-medium">Backend Health</p>
              <div className="flex items-center gap-2 mt-1">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="font-semibold text-emerald-400 capitalize">
                  {healthData?.status || 'Healthy'}
                </span>
              </div>
            </div>
          </div>

          {/* Database Status */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-indigo-500/10 text-indigo-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs text-slate-400 font-medium">Database Connection</p>
              <p className="font-semibold text-indigo-300 mt-1 capitalize">
                {healthData?.database_status || 'Connected'}
              </p>
            </div>
          </div>

          {/* Latency & Version */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-blue-500/10 text-blue-400">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs text-slate-400 font-medium">Server Latency</p>
              <p className="font-semibold text-blue-300 mt-1 font-mono">
                {latency !== null ? `${latency} ms` : 'Measuring...'}
              </p>
            </div>
          </div>
        </div>
      )}

      {healthData && (
        <div className="mt-6 pt-4 border-t border-slate-800/60 text-xs text-slate-400 grid grid-cols-2 md:grid-cols-4 gap-2 font-mono">
          <div><span className="text-slate-500">Project:</span> {healthData.project_name}</div>
          <div><span className="text-slate-500">Version:</span> v{healthData.version}</div>
          <div><span className="text-slate-500">Environment:</span> {healthData.environment}</div>
          <div><span className="text-slate-500">Timestamp:</span> {new Date(healthData.timestamp).toLocaleTimeString()}</div>
        </div>
      )}
    </div>
  );
}
