// ===================== State =====================
let RESULTS = null;
let selectedTicker = null;
let priceChart = null;
let backtestChart = null;
let importanceChart = null;

const REFRESH_COOLDOWN_MS = 45000;
let refreshLockedUntil = 0;

// ===================== Init =====================
document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  await loadResults();
  renderDashboard();
  renderDeepDiveTickerList(); // Populates ticker selection buttons
  document.getElementById('refresh-btn').addEventListener('click', handleRefresh);
});

async function loadResults() {
  const res = await fetch('data/results.json');
  if (!res.ok) throw new Error(`Failed to load results.json: ${res.status}`);
  RESULTS = await res.json();
  selectedTicker = Object.keys(RESULTS.tickers)[0] || null;
  document.getElementById('footer-generated').textContent = formatDate(RESULTS.generated_at);
}

// ===================== Tabs =====================
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      
      const tabId = btn.dataset.tab;
      document.getElementById(tabId).classList.add('active');

      // Redraw charts ONLY when the tab becomes visible so Chart.js gets real dimensions
      if (tabId === 'deepdive' && selectedTicker) {
        renderDeepDive(selectedTicker);
      } else if (tabId === 'performance') {
        renderPerformance();
      }
    });
  });
}

// ===================== Dashboard tab =====================
function renderDashboard() {
  const tickers = RESULTS.tickers;
  const count = Object.keys(tickers).length;
  document.getElementById('dashboard-subtitle').textContent =
    `${count} tickers · ${RESULTS.horizon_days}-day horizon · updated ${formatDate(RESULTS.generated_at)}`;

  const grid = document.getElementById('ticker-grid');
  grid.innerHTML = '';
  Object.entries(tickers).forEach(([symbol, t]) => {
    grid.appendChild(buildTickerCard(symbol, t));
  });
}

function buildTickerCard(symbol, t) {
  const card = document.createElement('div');
  card.className = 'ticker-card';
  card.tabIndex = 0;
  const isUp = t.prediction === 'UP';
  const pct = Math.round(t.probability_up * 100);

  card.innerHTML = `
    <div class="card-header">
      <div class="ticker-symbol">${symbol}</div>
      <div class="badge ${isUp ? 'up' : 'down'}">${t.prediction}</div>
    </div>
    <div class="ticker-price">$${t.last_price != null ? t.last_price.toFixed(2) : '—'}</div>
    <div class="ticker-prob">
      Probability (Up): ${pct}%
      <div class="prob-bar"><div class="prob-bar-fill ${isUp ? 'up' : 'down'}" style="width:${pct}%"></div></div>
    </div>
    <div class="ticker-asof">Data as of ${t.as_of_date}</div>
  `;
  card.addEventListener('click', () => jumpToDeepDive(symbol));
  card.addEventListener('keypress', (e) => { if (e.key === 'Enter') jumpToDeepDive(symbol); });
  return card;
}

function jumpToDeepDive(symbol) {
  selectedTicker = symbol;
  document.getElementById('ticker-select').value = symbol;
  document.querySelector('.tab-btn[data-tab="deepdive"]').click();
  renderDeepDive(symbol);
}

// ===================== Refresh button =====================
async function handleRefresh() {
  const btn = document.getElementById('refresh-btn');
  const status = document.getElementById('refresh-status');

  if (Date.now() < refreshLockedUntil) return;

  btn.disabled = true;
  status.textContent = 'Fetching fresh prices...';

  try {
    const res = await fetch('/api/refresh');
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const fresh = await res.json();

    Object.entries(fresh.tickers).forEach(([symbol, data]) => {
      if (RESULTS.tickers[symbol]) {
        Object.assign(RESULTS.tickers[symbol], data);
      }
    });
    RESULTS.generated_at = fresh.generated_at;

    renderDashboard();
    renderDeepDive(selectedTicker);
    document.getElementById('footer-generated').textContent = formatDate(fresh.generated_at);
    status.textContent = `Updated ${formatDate(fresh.generated_at)}. Insider data unchanged.`;
  } catch (err) {
    status.textContent = `Refresh failed: ${err.message}`;
  }

  refreshLockedUntil = Date.now() + REFRESH_COOLDOWN_MS;
  tickCooldown(btn, status);
}

function tickCooldown(btn, status) {
  const remaining = Math.ceil((refreshLockedUntil - Date.now()) / 1000);
  if (remaining <= 0) {
    btn.disabled = false;
    btn.textContent = 'Refresh Prices';
    return;
  }
  btn.textContent = `Refresh Prices (${remaining}s)`;
  setTimeout(() => tickCooldown(btn, status), 1000);
}

// ===================== Deep-Dive tab =====================
function renderDeepDiveTickerList() {
  const select = document.getElementById('ticker-select');
  select.innerHTML = '';
  Object.keys(RESULTS.tickers).forEach(symbol => {
    const opt = document.createElement('option');
    opt.value = symbol;
    opt.textContent = symbol;
    select.appendChild(opt);
  });
  select.addEventListener('change', (e) => {
    selectedTicker = e.target.value;
    renderDeepDive(selectedTicker);
  });
}

function renderDeepDive(symbol) {
  if (!symbol || !RESULTS.tickers[symbol]) return;
  const t = RESULTS.tickers[symbol];

  // Price chart
  const ctx = document.getElementById('price-chart').getContext('2d');
  if (priceChart) priceChart.destroy();
  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: t.price_history.map(p => p.date),
      datasets: [{
        label: symbol,
        data: t.price_history.map(p => p.close),
        borderColor: '#e8b94e',
        backgroundColor: 'rgba(232, 185, 78, 0.10)',
        fill: true,
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.35,
      }],
    },
    options: chartBaseOptions({ showLegend: false }),
  });

  // Feature table
  const featureTable = document.getElementById('feature-table');
  featureTable.innerHTML = '';
  Object.entries(t.features).forEach(([key, val]) => {
    const row = document.createElement('tr');
    row.innerHTML = `<td>${prettifyFeatureName(key)}</td><td>${formatFeatureValue(key, val)}</td>`;
    featureTable.appendChild(row);
  });

  // Insider table
  const body = document.getElementById('insider-table-body');
  body.innerHTML = '';
  t.insider_trades.forEach(trade => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${trade.date}</td>
      <td>${trade.insider}</td>
      <td>${trade.title}</td>
      <td class="${trade.type === 'Buy' ? 'tag-buy' : 'tag-sell'}">${trade.type}</td>
      <td>$${Number(trade.value).toLocaleString()}</td>
    `;
    body.appendChild(row);
  });
}

function prettifyFeatureName(key) {
  const map = {
    return_1d: '1-Day Return', return_5d: '5-Day Return',
    sma_10: '10-Day SMA', sma_30: '30-Day SMA',
    price_vs_sma30: 'Price vs 30-Day SMA', volatility_10d: '10-Day Volatility',
    insider_n_buys_30d: 'Insider Buys (30d)', insider_n_sells_30d: 'Insider Sells (30d)',
    insider_buy_sell_ratio_30d: 'Buy/Sell Ratio (30d)', insider_net_value_30d: 'Net Insider $ (30d)',
    insider_cluster_buy_flag: 'Cluster Buying',
  };
  return map[key] || key;
}

function formatFeatureValue(key, val) {
  if (val == null) return '—';
  if (key.includes('return') || key.includes('vs_sma') || key.includes('volatility')) return (val * 100).toFixed(2) + '%';
  if (key.includes('net_value')) return '$' + Math.round(val).toLocaleString();
  if (key === 'insider_cluster_buy_flag') return val ? 'Yes' : 'No';
  if (key.includes('sma')) return '$' + val.toFixed(2);
  return typeof val === 'number' ? val.toFixed(2) : val;
}

// ===================== Performance tab =====================
function renderPerformance() {
  const perf = RESULTS.model_performance;
  const m = perf.metrics;

  document.getElementById('performance-subtitle').textContent =
    `Evaluated on ${m.test_rows} test rows, split after ${m.train_test_split_date}`;
  document.getElementById('horizon-days').textContent = RESULTS.horizon_days;

  const metricsRow = document.getElementById('metrics-row');
  metricsRow.innerHTML = '';
  [
    ['Accuracy', m.accuracy], ['Precision', m.precision],
    ['Recall', m.recall], ['ROC-AUC', m.roc_auc],
  ].forEach(([label, val]) => {
    const box = document.createElement('div');
    box.className = 'metric-box';
    box.innerHTML = `<div class="metric-value">${(val * 100).toFixed(1)}<span style="font-size:1rem; color:var(--text-secondary)">%</span></div><div class="metric-label">${label}</div>`;
    metricsRow.appendChild(box);
  });

  // Backtest chart
  const bt = perf.backtest;
  const btCtx = document.getElementById('backtest-chart').getContext('2d');
  if (backtestChart) backtestChart.destroy();
  backtestChart = new Chart(btCtx, {
    type: 'line',
    data: {
      labels: bt.dates,
      datasets: [
        { label: 'Strategy', data: bt.strategy_cum_return.map(v => v * 100), borderColor: '#e8b94e', pointRadius: 0, borderWidth: 2, tension: 0.15 },
        { label: 'Buy & Hold', data: bt.buy_hold_cum_return.map(v => v * 100), borderColor: '#a3adc2', pointRadius: 0, borderWidth: 2, borderDash: [5, 4], tension: 0.15 },
      ],
    },
    options: chartBaseOptions({ showLegend: true, yLabel: '%' }),
  });

  // Feature importance chart
  const fi = perf.feature_importances;
  const sorted = Object.entries(fi).sort((a, b) => b[1] - a[1]);
  const impCtx = document.getElementById('importance-chart').getContext('2d');
  if (importanceChart) importanceChart.destroy();
  importanceChart = new Chart(impCtx, {
    type: 'bar',
    data: {
      labels: sorted.map(([k]) => prettifyFeatureName(k)),
      datasets: [{ data: sorted.map(([, v]) => v), backgroundColor: '#e8b94e', borderRadius: 4 }],
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#a3adc2' } },
        y: { grid: { display: false }, ticks: { font: { family: 'Inter', size: 12 }, color: '#f2f4f9' } },
      },
    },
  });
}

function chartBaseOptions({ showLegend, yLabel }) {
  return {
    plugins: { legend: { display: !!showLegend, labels: { font: { family: 'Inter', size: 11 }, color: '#a3adc2', usePointStyle: true } } },
    scales: {
      x: { grid: { display: false }, ticks: { font: { family: 'JetBrains Mono', size: 9 }, color: '#6b7690', maxTicksLimit: 6 } },
      y: {
        grid: { color: 'rgba(255,255,255,0.08)' },
        border: { display: false },
        ticks: {
          font: { family: 'JetBrains Mono', size: 10 },
          color: '#a3adc2',
          callback: v => yLabel ? v + yLabel : v,
        },
      },
    },
  };
}

function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}
