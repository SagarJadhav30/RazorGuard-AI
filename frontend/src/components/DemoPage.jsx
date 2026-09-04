import { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import './DemoPage.css';
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  FileCode2,
  FileText,
  Layers,
  Play,
  RefreshCcw,
  RotateCcw,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  UserCheck,
  Zap,
} from 'lucide-react';

const API = '/api/v1';
const money = (v) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(v || 0);
const timeStr = (v) =>
  v ? new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeStyle: 'medium' }).format(new Date(v)) : 'Just now';

function Badge({ level }) {
  const lvl = (level || 'unknown').toLowerCase();
  return <span className={`risk-badge ${lvl}`}>{level || 'UNKNOWN'}</span>;
}

function RiskGauge({ score = 0, level }) {
  const angle = Math.min(Math.max(Number(score), 0), 100) * 1.8 - 90;
  return (
    <div className="risk-gauge">
      <div className="gauge-dial">
        <div className="gauge-ticks" />
        <div className="gauge-needle" style={{ transform: `translateX(-50%) rotate(${angle}deg)` }} />
        <div className="gauge-center">
          <strong>{score}</strong>
          <span>Risk Score</span>
        </div>
      </div>
      <div className="gauge-scale">
        <span>0 LOW</span>
        <span>50 REVIEW</span>
        <span>100 HIGH</span>
      </div>
      <Badge level={level} />
    </div>
  );
}

export default function DemoPage({ onOpenInvestigation }) {
  const [scenarios, setScenarios] = useState([]);
  const [activeScenarioId, setActiveScenarioId] = useState('scenario_1');
  const [executionResult, setExecutionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [notification, setNotification] = useState('');
  const [error, setError] = useState('');

  // Fetch scenarios list
  const loadScenarios = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/demo/scenarios`);
      if (res.data?.scenarios) {
        setScenarios(res.data.scenarios);
      }
    } catch (e) {
      console.warn('Using fallback scenario list', e);
    }
  }, []);

  // Execute scenario
  const runScenario = useCallback(async (id) => {
    setActiveScenarioId(id);
    setLoading(true);
    setError('');
    setNotification('');
    try {
      const res = await axios.post(`${API}/demo/execute/${id}`);
      setExecutionResult(res.data);
      setNotification(`Successfully executed ${res.data?.scenario?.title || id}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to execute demo scenario.');
    } finally {
      setLoading(false);
    }
  }, []);

  // Reset demo
  const handleReset = async () => {
    setResetting(true);
    setError('');
    setNotification('');
    try {
      const res = await axios.post(`${API}/demo/reset`);
      setNotification(res.data?.message || 'Demo environment reset to baseline state.');
      setExecutionResult(null);
      // Re-run scenario 1 as default after reset
      setTimeout(() => runScenario('scenario_1'), 600);
    } catch (err) {
      setError('Failed to reset demo state.');
    } finally {
      setResetting(false);
    }
  };

  useEffect(() => {
    loadScenarios();
    runScenario('scenario_1');
  }, [loadScenarios, runScenario]);

  const activeScenarioMeta = scenarios.find((s) => s.id === activeScenarioId) || executionResult?.scenario;
  const assessment = executionResult?.assessment || {};
  const input = executionResult?.input || {};
  const shapFactors = executionResult?.shap_factors || { risk_factors: [], protective_factors: [] };
  const llm = executionResult?.llm_explanation || {};
  const audit = executionResult?.audit_event || {};

  return (
    <div className="demo-container">
      {/* Page Header */}
      <div className="page-intro">
        <div>
          <div className="eyebrow">Interactive Hackathon Hub</div>
          <h1>Live Demo Mode &amp; Explainability Inspector</h1>
          <p>
            Execute controlled synthetic fraud scenarios in real time. Observe how ML probabilities, deterministic
            policies, SHAP attribution, and GenAI narratives combine with zero black-box obscurity.
          </p>
        </div>
        <div className="demo-header-actions">
          <button
            className="demo-reset-btn"
            disabled={resetting || loading}
            onClick={handleReset}
            title="Purge demo test records and return to pristine state"
          >
            <RotateCcw size={14} className={resetting ? 'spin' : ''} />
            {resetting ? 'Resetting Baseline...' : 'Reset Demo State'}
          </button>
        </div>
      </div>

      {/* Notifications */}
      {notification && (
        <div className="success-banner">
          <CheckCircle2 size={16} />
          {notification}
        </div>
      )}
      {error && (
        <div className="state-banner">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Pipeline Visual Ribbon */}
      <div className="pipeline-ribbon">
        <div className="pipeline-step active">
          <span className="step-num">1</span>
          <span>Synthetic Input</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step active">
          <span className="step-num">2</span>
          <span>Feature Engineering</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step active">
          <span className="step-num">3</span>
          <span>ML Risk Model</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step active">
          <span className="step-num">4</span>
          <span>Policy Rules</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step active">
          <span className="step-num">5</span>
          <span>SHAP Attribution</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step active">
          <span className="step-num">6</span>
          <span>GenAI Narrative</span>
        </div>
        <span className="pipeline-arrow">&rarr;</span>
        <div className="pipeline-step active">
          <span className="step-num">7</span>
          <span>Audit Log</span>
        </div>
      </div>

      {/* Scenario Selection Grid (4 Scenarios) */}
      <section>
        <div className="panel-heading" style={{ marginBottom: 12 }}>
          <div>
            <h3>Pre-Configured Hackathon Scenarios</h3>
            <p>Select a scenario to execute live inference against the trained model &amp; rule engine</p>
          </div>
        </div>

        <div className="scenarios-grid">
          {/* Scenario 1 */}
          <div
            className={`scenario-card ${activeScenarioId === 'scenario_1' ? 'active' : ''}`}
            onClick={() => runScenario('scenario_1')}
          >
            <div className="scenario-card-header">
              <span className="scenario-card-badge badge-low">Scenario 1</span>
              <Badge level="LOW" />
            </div>
            <div>
              <div className="scenario-card-title">Normal Returning Customer</div>
              <div className="scenario-card-desc">Established account, domestic IP, consistent purchase volume</div>
            </div>
            <div className="scenario-card-meta">
              <span>Expected: <strong>APPROVE</strong></span>
              <span>Amount: <strong>$42.50</strong></span>
            </div>
            <button className="scenario-run-btn" disabled={loading && activeScenarioId === 'scenario_1'}>
              {loading && activeScenarioId === 'scenario_1' ? (
                <RefreshCcw size={13} className="spin" />
              ) : (
                <Play size={13} />
              )}
              {loading && activeScenarioId === 'scenario_1' ? 'Evaluating...' : 'Run Scenario 1'}
            </button>
          </div>

          {/* Scenario 2 */}
          <div
            className={`scenario-card ${activeScenarioId === 'scenario_2' ? 'active' : ''}`}
            onClick={() => runScenario('scenario_2')}
          >
            <div className="scenario-card-header">
              <span className="scenario-card-badge badge-medium">Scenario 2</span>
              <Badge level="MEDIUM" />
            </div>
            <div>
              <div className="scenario-card-title">New Customer + Unusual Amount</div>
              <div className="scenario-card-desc">5-day old account, $280 purchase (9x higher than $30 avg)</div>
            </div>
            <div className="scenario-card-meta">
              <span>Expected: <strong>REVIEW</strong></span>
              <span>Amount: <strong>$280.00</strong></span>
            </div>
            <button className="scenario-run-btn" disabled={loading && activeScenarioId === 'scenario_2'}>
              {loading && activeScenarioId === 'scenario_2' ? (
                <RefreshCcw size={13} className="spin" />
              ) : (
                <Play size={13} />
              )}
              {loading && activeScenarioId === 'scenario_2' ? 'Evaluating...' : 'Run Scenario 2'}
            </button>
          </div>

          {/* Scenario 3 */}
          <div
            className={`scenario-card ${activeScenarioId === 'scenario_3' ? 'active' : ''}`}
            onClick={() => runScenario('scenario_3')}
          >
            <div className="scenario-card-header">
              <span className="scenario-card-badge badge-high">Scenario 3</span>
              <Badge level="HIGH" />
            </div>
            <div>
              <div className="scenario-card-title">Retries + Device Reuse Attack</div>
              <div className="scenario-card-desc">3 failed attempts, device shared across 8 accounts, velocity 88</div>
            </div>
            <div className="scenario-card-meta">
              <span>Expected: <strong>BLOCK</strong></span>
              <span>Amount: <strong>$1,250.00</strong></span>
            </div>
            <button className="scenario-run-btn" disabled={loading && activeScenarioId === 'scenario_3'}>
              {loading && activeScenarioId === 'scenario_3' ? (
                <RefreshCcw size={13} className="spin" />
              ) : (
                <Play size={13} />
              )}
              {loading && activeScenarioId === 'scenario_3' ? 'Evaluating...' : 'Run Scenario 3'}
            </button>
          </div>

          {/* Scenario 4 */}
          <div
            className={`scenario-card ${activeScenarioId === 'scenario_4' ? 'active' : ''}`}
            onClick={() => runScenario('scenario_4')}
          >
            <div className="scenario-card-header">
              <span className="scenario-card-badge badge-low">Scenario 4</span>
              <Badge level="LOW" />
            </div>
            <div>
              <div className="scenario-card-title">VIP Customer High-Value ($2,450)</div>
              <div className="scenario-card-desc">Demonstrates high ticket size alone != fraud with verified loyalty</div>
            </div>
            <div className="scenario-card-meta">
              <span>Expected: <strong>APPROVE</strong></span>
              <span>Amount: <strong>$2,450.00</strong></span>
            </div>
            <button className="scenario-run-btn" disabled={loading && activeScenarioId === 'scenario_4'}>
              {loading && activeScenarioId === 'scenario_4' ? (
                <RefreshCcw size={13} className="spin" />
              ) : (
                <Play size={13} />
              )}
              {loading && activeScenarioId === 'scenario_4' ? 'Evaluating...' : 'Run Scenario 4'}
            </button>
          </div>
        </div>
      </section>

      {/* Results Explorer (Live Execution Outputs) */}
      {executionResult && (
        <div className="demo-results-zone">
          {/* Headline & Focus Banner */}
          <div className="results-headline-banner">
            <div className="results-headline-text">
              <h2>{activeScenarioMeta?.title || executionResult.scenario?.title}</h2>
              <p>{activeScenarioMeta?.headline || executionResult.scenario?.headline}</p>
              <div className="demo-focus-callout">
                <strong>Demonstration Focus &amp; Value</strong>
                <p>{activeScenarioMeta?.demonstration_focus || executionResult.scenario?.demonstration_focus}</p>
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div className={`decision large ${assessment.decision?.toLowerCase()}`}>
                <i />
                {assessment.decision}
              </div>
            </div>
          </div>

          {/* 2-Column: 1. Input Attributes | 2. ML & Decision Outcome */}
          <div className="demo-top-grid">
            {/* Input Payload */}
            <div className="input-payload-card">
              <div className="panel-heading">
                <div>
                  <h3>1. Transaction Input Attributes</h3>
                  <p>Reproducible synthetic payload passed into feature pipeline</p>
                </div>
                <FileCode2 size={18} className="accent-icon" />
              </div>
              <div className="input-attributes-grid">
                <div className="input-attr-item">
                  <span>Transaction ID</span>
                  <strong>{input.transaction_id}</strong>
                </div>
                <div className="input-attr-item">
                  <span>Customer ID</span>
                  <strong>{input.customer_id}</strong>
                </div>
                <div className="input-attr-item">
                  <span>Amount &amp; Currency</span>
                  <strong>
                    {money(input.amount)} {input.currency}
                  </strong>
                </div>
                <div className="input-attr-item">
                  <span>Merchant Category</span>
                  <strong style={{ textTransform: 'capitalize' }}>{input.merchant_category}</strong>
                </div>
                <div className="input-attr-item">
                  <span>Account Age</span>
                  <strong>{input.account_age_days} days</strong>
                </div>
                <div className="input-attr-item">
                  <span>Historical Tx Count</span>
                  <strong>{input.customer_transaction_count} orders</strong>
                </div>
                <div className="input-attr-item">
                  <span>Failed Attempts</span>
                  <strong style={{ color: input.failed_payment_count > 0 ? '#ff6577' : '#39d98a' }}>
                    {input.failed_payment_count} failed
                  </strong>
                </div>
                <div className="input-attr-item">
                  <span>Device Reuse Across Accounts</span>
                  <strong style={{ color: input.device_reuse_count > 3 ? '#ff6577' : '#e2edf7' }}>
                    {input.device_reuse_count} {input.device_reuse_count > 1 ? 'accounts' : 'account'}
                  </strong>
                </div>
                <div className="input-attr-item">
                  <span>24h Velocity Score</span>
                  <strong>{input.velocity_score} / 100</strong>
                </div>
                <div className="input-attr-item">
                  <span>Country &amp; Location Match</span>
                  <strong>
                    {input.country} ({input.billing_shipping_match === 1 ? 'Billing Match' : 'Mismatch'})
                  </strong>
                </div>
              </div>
            </div>

            {/* ML & Risk Engine Decision Card */}
            <div className="decision-card-demo">
              <div className="panel-heading">
                <div>
                  <h3>2. ML Prediction &amp; Decision</h3>
                  <p>Calibrated risk scoring and deterministic policy overrides</p>
                </div>
                <Cpu size={18} className="accent-icon" />
              </div>

              <RiskGauge score={assessment.final_risk_score} level={assessment.risk_level} />

              <div className="decision-meta" style={{ marginTop: 14 }}>
                <span>
                  Fraud Probability: <b>{((assessment.fraud_probability || 0) * 100).toFixed(1)}%</b>
                </span>
                <span>
                  ML Base Score: <b>{assessment.ml_risk_score}</b>
                </span>
                <span>
                  Anomaly Score: <b>{assessment.anomaly_score}</b>
                </span>
              </div>

              {assessment.triggered_rules?.length > 0 && (
                <div style={{ marginTop: 14 }}>
                  <span className="eyebrow" style={{ color: '#f4b860' }}>
                    Triggered Policy Rules
                  </span>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                    {assessment.triggered_rules.map((rule) => (
                      <span
                        key={rule}
                        style={{
                          background: 'rgba(244, 184, 96, 0.15)',
                          color: '#f4b860',
                          border: '1px solid rgba(244, 184, 96, 0.3)',
                          padding: '3px 8px',
                          borderRadius: 4,
                          fontSize: 10,
                          fontFamily: 'DM Mono, monospace',
                        }}
                      >
                        {rule}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 3. SHAP Feature Attribution (Diverging Waterfall Breakdown) */}
          <section className="explanation-panel" style={{ marginTop: 0 }}>
            <div className="panel-heading">
              <div>
                <h3>3. SHAP Local Feature Attribution (Explainable AI)</h3>
                <p>Mathematical evidence isolating exact positive (risk-increasing) and negative (protective) drivers</p>
              </div>
              <Layers size={18} className="accent-icon" />
            </div>

            <div className="shap-factors-container">
              {/* Risk Increasing Factors */}
              <div className="shap-column">
                <div className="shap-column-header risk">
                  <TrendingUp size={15} />
                  <span>Risk-Increasing Drivers (+SHAP)</span>
                </div>
                {shapFactors.risk_factors?.length > 0 ? (
                  shapFactors.risk_factors.slice(0, 4).map((f) => (
                    <div className="shap-bar-item" key={f.feature}>
                      <div className="shap-bar-meta">
                        <strong>{f.feature_name_human}</strong>
                        <span className="risk">+{f.shap_value.toFixed(3)}</span>
                      </div>
                      <div className="shap-bar-track">
                        <div
                          className="shap-bar-fill risk"
                          style={{
                            width: `${Math.min(Math.max(Math.abs(f.shap_value) * 20, 10), 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ fontSize: 11, color: '#698399', padding: '12px 0' }}>
                    No significant risk-increasing factors detected.
                  </div>
                )}
              </div>

              {/* Protective Factors */}
              <div className="shap-column">
                <div className="shap-column-header protective">
                  <TrendingDown size={15} />
                  <span>Protective Trust Signals (-SHAP)</span>
                </div>
                {shapFactors.protective_factors?.length > 0 ? (
                  shapFactors.protective_factors.slice(0, 4).map((f) => (
                    <div className="shap-bar-item" key={f.feature}>
                      <div className="shap-bar-meta">
                        <strong>{f.feature_name_human}</strong>
                        <span className="protective">{f.shap_value.toFixed(3)}</span>
                      </div>
                      <div className="shap-bar-track">
                        <div
                          className="shap-bar-fill protective"
                          style={{
                            width: `${Math.min(Math.max(Math.abs(f.shap_value) * 20, 10), 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ fontSize: 11, color: '#698399', padding: '12px 0' }}>
                    No significant protective factors detected.
                  </div>
                )}
              </div>
            </div>
          </section>

          {/* 4. GenAI Risk Analyst Narrative & 5. Audit Event */}
          <div className="investigation-lower" style={{ marginTop: 0 }}>
            {/* GenAI Explanation */}
            <section className="explanation-panel">
              <div className="panel-heading">
                <div>
                  <h3>4. GenAI Risk Analyst Incident Report</h3>
                  <p>Natural language explanation grounded in SHAP mathematical evidence</p>
                </div>
                <Sparkles size={18} className="accent-icon" style={{ color: '#5ba7ff' }} />
              </div>
              <div className="ai-copy">
                <strong>{llm.summary_headline || 'Explanation generated'}</strong>
                <p>{llm.why_flagged || 'Evaluation completed based on features.'}</p>
                {llm.recommended_analyst_action && (
                  <div
                    style={{
                      background: 'rgba(91, 167, 255, 0.08)',
                      borderLeft: '2px solid #5ba7ff',
                      padding: '8px 12px',
                      margin: '10px 0',
                      borderRadius: '0 4px 4px 0',
                    }}
                  >
                    <span style={{ fontSize: 10, color: '#8ec3f8', textTransform: 'uppercase', fontWeight: 700 }}>
                      Recommended Action:
                    </span>
                    <p style={{ margin: '4px 0 0', fontSize: 12, color: '#d8e5f2' }}>
                      {llm.recommended_analyst_action}
                    </p>
                  </div>
                )}
                <span>
                  Provider: {llm.provider_used || 'Gemini / Deterministic Fallback'} &bull; Zero Execution Authority
                  Enforced
                </span>
              </div>
            </section>

            {/* Audit Event Entry */}
            <section className="timeline-panel">
              <div className="panel-heading">
                <div>
                  <h3>5. Immutable Audit Event</h3>
                  <p>Forensic compliance record logged to database</p>
                </div>
                <Shield size={18} className="accent-icon" />
              </div>
              <div className="timeline">
                <div className="timeline-item">
                  <span className={`timeline-dot ${audit.decision?.toLowerCase()}`} />
                  <div>
                    <div className="timeline-title">
                      <strong>{audit.action || 'DEMO_ASSESSMENT'}</strong>
                      <span>{timeStr(audit.timestamp)}</span>
                    </div>
                    <p>{audit.reason || 'Scenario evaluated through live demo engine'}</p>
                    <small>
                      Actor: {audit.actor || 'DEMO_CONTROLLER'} &bull; Request: {audit.request_id} &bull; Score:{' '}
                      {audit.risk_score}
                    </small>
                  </div>
                  <Badge level={audit.decision} />
                </div>
              </div>

              {onOpenInvestigation && (
                <div style={{ marginTop: 14, textAlign: 'right' }}>
                  <button
                    className="secondary-button"
                    style={{ fontSize: 11, padding: '7px 12px' }}
                    onClick={() => onOpenInvestigation(input.transaction_id)}
                  >
                    Open in Full Investigation Page &rarr;
                  </button>
                </div>
              )}
            </section>
          </div>
        </div>
      )}
    </div>
  );
}
