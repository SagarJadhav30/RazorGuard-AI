import { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import {
  ResponsiveContainer, PieChart, Pie, Cell,
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip
} from 'recharts';
import {
  Activity, AlertTriangle, ArrowRight, BarChart3, Bell, ChevronRight,
  DollarSign, Eye, FileText, LayoutDashboard, Menu, RefreshCw,
  Shield, ShieldAlert, Sparkles, TrendingUp
} from 'lucide-react';
import InvestigationPage from './components/InvestigationPage';
import ModelPerformancePage from './components/ModelPerformancePage';
import FinancialImpactPage from './components/FinancialImpactPage';
import DemoPage from './components/DemoPage';

const API = '/api/v1';
const PIE_COLORS = ['#39d98a', '#f4b860', '#ff6577'];
const money = (v) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v || 0);

const NAV = [
  ['dashboard', 'Dashboard', LayoutDashboard],
  ['demo', 'Live Demo Hub', Sparkles],
  ['transactions', 'Transactions', FileText],
  ['analytics', 'Risk Analytics', BarChart3],
  ['performance', 'Model Performance', TrendingUp],
  ['audit', 'Audit Trail', Shield],
  ['impact', 'Financial Impact', DollarSign],
];

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      {payload.map((p) => (
        <span key={p.dataKey}><i style={{ background: p.color }} />{p.name}: {p.value}</span>
      ))}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [selectedId, setSelectedId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [data, setData] = useState({
    summary: null, transactions: [], metrics: null, audit: [], health: null,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const [sRes, tRes, mRes, aRes, hRes] = await Promise.allSettled([
          axios.get(`${API}/risk/summary`),
          axios.get(`${API}/transactions?limit=500`),
          axios.get(`${API}/model/metrics`),
          axios.get(`${API}/audit?limit=100`),
          axios.get(`${API}/health`),
        ]);
        setData({
          summary: sRes.status === 'fulfilled' ? sRes.value.data : null,
          transactions: tRes.status === 'fulfilled' ? (tRes.value.data.items || tRes.value.data || []) : [],
          metrics: mRes.status === 'fulfilled' ? mRes.value.data : null,
          audit: aRes.status === 'fulfilled' ? (aRes.value.data.items || aRes.value.data || []) : [],
          health: hRes.status === 'fulfilled' ? hRes.value.data : null,
        });
      } catch { /* individual field fallbacks above */ }
      setLoading(false);
    })();
  }, []);

  const open = useCallback((txId) => {
    setSelectedId(txId);
    setPage('investigation');
  }, []);

  /* ── derived dashboard stats ── */
  const txs = data.transactions || [];
  const high = txs.filter((t) => (t.assessment?.risk_level || '').toUpperCase() === 'HIGH').length;
  const blocked = txs.filter((t) => t.assessment?.decision === 'BLOCK').length;
  const reviewing = txs.filter((t) => t.assessment?.decision === 'REVIEW').length;
  const cm = data.metrics?.confusion_matrix || {};
  const fpCount = Number(cm.false_positives || 0);
  const tnCount = Number(cm.true_negatives || 0);
  const fpr = tnCount + fpCount > 0 ? fpCount / (tnCount + fpCount) : 0;
  const saved = data.metrics?.cost_analysis?.net_financial_savings_usd || 0;
  const healthOk = data.health?.status === 'healthy';

  /* ── chart datasets ── */
  const riskDist = [
    { name: 'Low', value: txs.filter((t) => (t.assessment?.risk_level || '').toUpperCase() === 'LOW').length },
    { name: 'Medium', value: txs.filter((t) => (t.assessment?.risk_level || '').toUpperCase() === 'MEDIUM').length },
    { name: 'High', value: high },
  ];
  const fraudLegit = [
    { name: 'Legitimate', value: Math.max(txs.length - blocked, 0) },
    { name: 'Fraud', value: blocked },
  ];
  const volumeData = Array.from({ length: 24 }, (_, h) => ({
    hour: `${h}:00`,
    count: txs.filter((t) => new Date(t.created_at).getHours() === h).length,
  }));
  const fraudTrend = Array.from({ length: 24 }, (_, h) => ({
    hour: `${h}:00`,
    fraud: txs.filter((t) => t.assessment?.decision === 'BLOCK' && new Date(t.created_at).getHours() === h).length,
  }));
  const decisionData = [
    { name: 'Approve', value: txs.filter((t) => t.assessment?.decision === 'APPROVE').length },
    { name: 'Review', value: reviewing },
    { name: 'Block', value: blocked },
  ];

  const currentNav = NAV.find(([key]) => key === page) || NAV[0];
  const pageLabel = page === 'investigation' ? 'Transaction Investigation' : (currentNav?.[1] || 'Dashboard');

  /* ── page content router ── */
  const content = (() => {
    switch (page) {
      /* ── Live Hackathon Demo (Controlled Scenarios) ── */
      case 'demo':
        return <DemoPage onOpenInvestigation={open} />;

      /* ── Investigation (Phase 12) ── */
      case 'investigation':
        return <InvestigationPage id={selectedId} back={() => setPage('transactions')} />;

      /* ── Model Performance (Phase 13) ── */
      case 'performance':
        return <ModelPerformancePage metrics={data.metrics} />;

      /* ── Financial Impact (Phase 14) ── */
      case 'impact':
        return <FinancialImpactPage />;

      /* ── Transactions ── */
      case 'transactions':
        return (
          <>
            <div className="page-intro">
              <div>
                <div className="eyebrow">Operations / Transactions</div>
                <h1>Transactions</h1>
                <p>Click any row to open the Transaction Investigation view.</p>
              </div>
            </div>
            {!data.transactions.length ? (
              <div className="empty-state">
                <FileText size={20} />
                <strong>No backend data available</strong>
                <span>Start the backend and send test transactions.</span>
              </div>
            ) : (
              <div className="table-panel">
                <div className="table-toolbar">
                  <span>{txs.length} transactions loaded</span>
                  <span className="toolbar-muted">Click a row to investigate</span>
                </div>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Transaction ID</th>
                        <th>Amount</th>
                        <th>Country</th>
                        <th>Risk Level</th>
                        <th>Decision</th>
                        <th>Fraud Prob.</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {txs.slice(0, 100).map((item) => (
                        <tr key={item.transaction_id} onClick={() => open(item.transaction_id)}>
                          <td className="mono">{item.transaction_id}</td>
                          <td>{money(item.amount)}</td>
                          <td>{item.country || 'Not supplied by API'}</td>
                          <td>
                            <span className={`risk-badge ${(item.assessment?.risk_level || 'unknown').toLowerCase()}`}>
                              {item.assessment?.risk_level || 'N/A'}
                            </span>
                          </td>
                          <td>
                            <span className={`decision ${(item.assessment?.decision || '').toLowerCase()}`}>
                              <i />{item.assessment?.decision || 'N/A'}
                            </span>
                          </td>
                          <td className="mono">{((item.assessment?.fraud_probability || 0) * 100).toFixed(1)}%</td>
                          <td className="row-arrow"><ArrowRight size={14} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        );

      /* ── Risk Analytics ── */
      case 'analytics':
        return (
          <>
            <div className="page-intro">
              <div>
                <div className="eyebrow">Intelligence / Risk Analytics</div>
                <h1>Risk Analytics</h1>
                <p>Aggregated risk distribution and pattern analysis from live backend data.</p>
              </div>
            </div>
            {!txs.length ? (
              <div className="empty-state">
                <BarChart3 size={20} />
                <strong>No backend data available</strong>
                <span>Load transactions to populate analytics.</span>
              </div>
            ) : (
              <div className="chart-grid">
                <div className="chart-panel">
                  <div className="panel-heading"><div><h3>Risk distribution</h3><p>Breakdown by risk tier</p></div></div>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={riskDist} dataKey="value" cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={2}>
                        {riskDist.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                      </Pie>
                      <Tooltip content={<ChartTip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="chart-panel">
                  <div className="panel-heading"><div><h3>Decision distribution</h3><p>Approve / Review / Block</p></div></div>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={decisionData} dataKey="value" cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={2}>
                        {decisionData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                      </Pie>
                      <Tooltip content={<ChartTip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="chart-panel wide">
                  <div className="panel-heading"><div><h3>Fraud trend</h3><p>Blocked transactions over time</p></div></div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={fraudTrend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1d3042" />
                      <XAxis dataKey="hour" tick={{ fill: '#657b91', fontSize: 9 }} />
                      <YAxis tick={{ fill: '#657b91', fontSize: 9 }} />
                      <Tooltip content={<ChartTip />} />
                      <Line type="monotone" dataKey="fraud" stroke="#ff6577" dot={false} name="Fraud" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </>
        );

      /* ── Audit Trail ── */
      case 'audit':
        return (
          <>
            <div className="page-intro">
              <div>
                <div className="eyebrow">Compliance / Audit Trail</div>
                <h1>Audit Trail</h1>
                <p>Complete append-only decision log from the backend.</p>
              </div>
            </div>
            {!data.audit.length ? (
              <div className="empty-state">
                <Shield size={20} />
                <strong>No audit records available</strong>
                <span>Process transactions to generate audit events.</span>
              </div>
            ) : (
              <div className="audit-list">
                {data.audit.map((event) => (
                  <div className="audit-item" key={event.id}>
                    <div className="audit-icon"><Shield size={16} /></div>
                    <div className="audit-main">
                      <div><strong>{event.action || event.action_taken}</strong></div>
                      <p>{event.reason || 'Automated decision'}</p>
                      <div className="audit-meta">
                        <span>Actor: {event.actor || event.performed_by}</span>
                        <span>Model: {event.model_version || 'Unknown'}</span>
                      </div>
                    </div>
                    <div className="audit-decision">
                      <span className={`risk-badge ${(event.decision || '').toLowerCase()}`}>
                        {event.decision || 'N/A'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        );

      /* ── Dashboard (default) ── */
      default:
        return (
          <>
            <div className="page-intro">
              <div>
                <div className="eyebrow">Overview / Risk Intelligence</div>
                <h1>Dashboard</h1>
                <p>Real-time operational overview from live backend endpoints.</p>
              </div>
              <button className="secondary-button" onClick={() => window.location.reload()}>
                <RefreshCw size={15} />Refresh
              </button>
            </div>
            {loading ? (
              <div className="loading-state"><RefreshCw className="spin" size={18} />Loading dashboard data...</div>
            ) : (
              <>
                {/* ── Hackathon Demo Quick Launcher Banner ── */}
                <div
                  style={{
                    background: 'linear-gradient(135deg, #10253b, #0c1825)',
                    border: '1px solid #234768',
                    borderRadius: 8,
                    padding: '14px 20px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 20,
                    boxShadow: '0 4px 18px rgba(0,0,0,0.3)'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                    <div
                      style={{
                        background: '#194268',
                        padding: 10,
                        borderRadius: 8,
                        color: '#5ba7ff',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      <Sparkles size={20} />
                    </div>
                    <div>
                      <strong style={{ color: '#f0f7fc', fontSize: 13, display: 'block', marginBottom: 2 }}>
                        Live Hackathon Demo Mode Ready
                      </strong>
                      <span style={{ color: '#8da4b8', fontSize: 11 }}>
                        Test 4 controlled scenarios (Low Risk Approve, New Customer Review, Attack Block, VIP High-Value) with real-time SHAP &amp; GenAI explainability.
                      </span>
                    </div>
                  </div>
                  <button
                    className="primary-button"
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 8,
                      background: 'linear-gradient(135deg, #25679f, #1a4c78)',
                      color: '#ffffff',
                      border: '1px solid #5ba7ff',
                      padding: '8px 16px',
                      borderRadius: 6,
                      fontSize: 12,
                      fontWeight: 700,
                      cursor: 'pointer',
                      whiteSpace: 'nowrap'
                    }}
                    onClick={() => setPage('demo')}
                  >
                    <Sparkles size={14} />
                    Launch Live Demo &rarr;
                  </button>
                </div>

                {/* ── Metric cards ── */}
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-top">
                      <div className="metric-icon"><Activity size={16} /></div>
                      <span className="metric-label">Total transactions</span>
                    </div>
                    <div className="metric-value">{txs.length || 0}</div>
                    <div className="metric-note">{txs.length ? 'From live backend' : 'No backend data available'}</div>
                  </div>
                  <div className="metric-card red">
                    <div className="metric-top">
                      <div className="metric-icon"><ShieldAlert size={16} /></div>
                      <span className="metric-label">High risk transactions</span>
                    </div>
                    <div className="metric-value">{high}</div>
                  </div>
                  <div className="metric-card orange">
                    <div className="metric-top">
                      <div className="metric-icon"><AlertTriangle size={16} /></div>
                      <span className="metric-label">Fraud detected</span>
                    </div>
                    <div className="metric-value">{blocked}</div>
                  </div>
                  <div className="metric-card yellow">
                    <div className="metric-top">
                      <div className="metric-icon"><Eye size={16} /></div>
                      <span className="metric-label">Review queue</span>
                    </div>
                    <div className="metric-value">{reviewing}</div>
                  </div>
                  <div className="metric-card green">
                    <div className="metric-top">
                      <div className="metric-icon"><DollarSign size={16} /></div>
                      <span className="metric-label">Estimated loss prevented</span>
                    </div>
                    <div className="metric-value">{data.metrics ? money(saved) : 'No model metrics available'}</div>
                  </div>
                  <div className="metric-card purple">
                    <div className="metric-top">
                      <div className="metric-icon"><TrendingUp size={16} /></div>
                      <span className="metric-label">False positive rate</span>
                    </div>
                    <div className="metric-value">{data.metrics ? `${(fpr * 100).toFixed(2)}%` : 'No model metrics available'}</div>
                  </div>
                </div>

                {/* ── Charts ── */}
                <div className="chart-grid">
                  <div className="chart-panel">
                    <div className="panel-heading"><div><h3>Risk distribution</h3><p>Breakdown across risk tiers</p></div></div>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie data={riskDist} dataKey="value" cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={2}>
                          {riskDist.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                        </Pie>
                        <Tooltip content={<ChartTip />} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-panel">
                    <div className="panel-heading"><div><h3>Fraud vs legitimate</h3><p>Proportion based on backend decisions</p></div></div>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie data={fraudLegit} dataKey="value" cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={2}>
                          {fraudLegit.map((_, i) => <Cell key={i} fill={['#5ba7ff', '#ff6577'][i]} />)}
                        </Pie>
                        <Tooltip content={<ChartTip />} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-panel">
                    <div className="panel-heading"><div><h3>Transaction volume</h3><p>Hourly volume distribution</p></div></div>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={volumeData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1d3042" />
                        <XAxis dataKey="hour" tick={{ fill: '#657b91', fontSize: 9 }} />
                        <YAxis tick={{ fill: '#657b91', fontSize: 9 }} />
                        <Tooltip content={<ChartTip />} />
                        <Area type="monotone" dataKey="count" stroke="#5ba7ff" fill="#5ba7ff22" name="Volume" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-panel">
                    <div className="panel-heading"><div><h3>Fraud trend</h3><p>Blocked transactions by hour</p></div></div>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={fraudTrend}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1d3042" />
                        <XAxis dataKey="hour" tick={{ fill: '#657b91', fontSize: 9 }} />
                        <YAxis tick={{ fill: '#657b91', fontSize: 9 }} />
                        <Tooltip content={<ChartTip />} />
                        <Line type="monotone" dataKey="fraud" stroke="#ff6577" dot={false} name="Fraud" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-panel wide">
                    <div className="panel-heading"><div><h3>Decision distribution</h3><p>Approve / Review / Block breakdown</p></div></div>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie data={decisionData} dataKey="value" cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={2}>
                          {decisionData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                        </Pie>
                        <Tooltip content={<ChartTip />} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}
          </>
        );
    }
  })();

  return (
    <div className="app-shell">
      {sidebarOpen && <button className="backdrop" onClick={() => setSidebarOpen(false)} />}

      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="brand">
          <div className="brand-mark"><Shield size={20} /></div>
          <div>RAZOR<b>GUARD</b><small>AI Risk Manager</small></div>
        </div>
        <div className="workspace">
          <span className={`status-dot ${healthOk ? '' : 'offline'}`} />
          <span>{healthOk ? 'Backend connected' : 'Connecting...'}</span>
          <span>{data.health?.environment || 'dev'}</span>
        </div>
        <nav>
          {NAV.map(([key, label, Icon]) => (
            <button key={key} className={page === key ? 'active' : ''} onClick={() => setPage(key)}>
              <Icon size={16} />{label}
              {key === 'transactions' && txs.length > 0 && <em>{txs.length}</em>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="api-health">
            <span className={`status-dot ${healthOk ? '' : 'offline'}`} />
            <div>
              <strong>API {healthOk ? 'Connected' : 'Offline'}</strong>
              <small>{data.health?.timestamp ? new Date(data.health.timestamp).toLocaleTimeString() : 'Not available'}</small>
            </div>
          </div>
          <div className="version">RazorGuard AI <span>v{data.health?.version || '1.0.0'}</span></div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="main-content">
        <div className="topbar">
          <div className="crumb">
            <button className="mobile-menu icon-button" onClick={() => setSidebarOpen(true)}><Menu size={18} /></button>
            <span>RazorGuard</span>
            <ChevronRight size={12} />
            <strong>{pageLabel}</strong>
          </div>
          <div className="top-actions">
            <div className="live-label"><span className="status-dot" />Live</div>
            <button className="icon-button notification"><Bell size={16} /><i /></button>
            <div className="avatar">SG</div>
          </div>
        </div>
        <div className="page-wrap">{content}</div>
      </div>
    </div>
  );
}
