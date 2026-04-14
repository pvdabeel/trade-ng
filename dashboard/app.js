const API = window.location.origin + '/api';
const REFRESH_MS = 15000;

let equityChart = null;
let drawdownChart = null;
let currencyChart = null;
let fxEvolutionChart = null;
let performanceChart = null;
let benchmarkChart = null;
let strategyChart = null;
let confidenceChart = null;
let displayCurrency = 'USD';
let _cachedBlocklist = [];
let latestEurUsdRate = null;
let selectedHours = 1;

function $(sel) { return document.querySelector(sel); }

function fmtPrice(n) {
  if (n == null) return '--';
  if (n >= 1) return '$' + n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  if (n >= 0.01) return '$' + n.toFixed(4);
  if (n >= 0.0001) return '$' + n.toFixed(6);
  return '$' + n.toFixed(10);
}

function fmtUsd(n) {
  if (n == null) return '$--';
  return '$' + n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}

function fmtEur(n) {
  if (n == null) return '--';
  return '\u20AC' + n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}

function toDisplay(usdVal) {
  if (usdVal == null) return null;
  if (displayCurrency === 'EUR' && latestEurUsdRate > 0) return usdVal / latestEurUsdRate;
  return usdVal;
}
function fmtCurr(n) {
  const v = toDisplay(n);
  if (v == null) return displayCurrency === 'EUR' ? '\u20AC--' : '$--';
  const sym = displayCurrency === 'EUR' ? '\u20AC' : '$';
  return sym + v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
function currSym() { return displayCurrency === 'EUR' ? '\u20AC' : '$'; }

function pct(n) { return n != null ? n.toFixed(1) + '%' : '--%'; }

// ---- Trend tracking for top cards ----
const prevMetrics = {};

function setTrend(elId, currentVal, key, opts = {}) {
  const el = document.getElementById(elId);
  if (!el || currentVal == null) return;
  const prev = prevMetrics[key];
  prevMetrics[key] = currentVal;
  if (prev == null) { el.innerHTML = ''; return; }

  const diff = currentVal - prev;
  const threshold = opts.threshold || 0.005;
  if (Math.abs(diff) < threshold) {
    el.className = 'card-trend trend-flat';
    el.innerHTML = '<span class="trend-arrow">\u25B6</span> flat';
    return;
  }

  const up = diff > 0;
  const invert = opts.invert || false;
  const trendClass = (up !== invert) ? 'trend-up' : 'trend-down';
  const arrow = up ? '\u25B2' : '\u25BC';

  let label;
  if (opts.format === 'pct') {
    label = (up ? '+' : '') + diff.toFixed(2) + '%';
  } else if (opts.format === 'curr') {
    const sym = currSym();
    label = (up ? '+' + sym : '-' + sym) + Math.abs(diff).toFixed(2);
  } else if (opts.format === 'eur') {
    label = (up ? '+\u20AC' : '-\u20AC') + Math.abs(diff).toFixed(2);
  } else {
    label = (up ? '+$' : '-$') + Math.abs(diff).toFixed(2);
  }

  el.className = 'card-trend ' + trendClass;
  el.innerHTML = `<span class="trend-arrow">${arrow}</span> ${label}`;
}

function fmtSize(n) {
  if (n == null) return '--';
  if (n >= 1000) return n.toLocaleString('en-US', {maximumFractionDigits: 0});
  return n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 6});
}

async function fetchJSON(path) {
  try {
    const res = await fetch(API + path);
    if (!res.ok) throw new Error(res.statusText);
    return await res.json();
  } catch (e) {
    console.error('Fetch error:', path, e);
    return null;
  }
}

// ---- Portfolio + FX ----

async function updatePortfolio() {
  const data = await fetchJSON('/portfolio');
  if (!data) return;

  if (data.fx) {
    const fx = data.fx;
    latestEurUsdRate = fx.eur_usd_rate;
    $('#fx-rate').textContent = 'EUR/USD ' + fx.eur_usd_rate.toFixed(4);

    const expPct = (fx.usd_exposure_pct * 100).toFixed(0) + '%';
    const expEl = $('#usd-exposure');
    expEl.textContent = expPct;
    expEl.className = 'card-value ' + (fx.usd_exposure_pct <= 0.35 ? 'positive' : fx.usd_exposure_pct <= 0.50 ? '' : 'negative');

    const eurcUsd = fx.eurc_balance * fx.eur_usd_rate;
    const cashTotalUsd = fx.usdc_balance + eurcUsd;

    $('#cash-total').textContent = fmtCurr(cashTotalUsd);
    $('#cash-usdc-line').textContent = 'USDC: ' + fmtUsd(fx.usdc_balance);
    $('#cash-eurc-line').textContent = 'EURC: ' + fx.eurc_balance.toFixed(0) + ' (\u2248' + fmtUsd(eurcUsd) + ')';

    setTrend('trend-cash-total', toDisplay(cashTotalUsd), 'cash_total', { format: 'curr' });
    setTrend('trend-exposure', fx.usd_exposure_pct * 100, 'usd_exposure', { format: 'pct', invert: true });

    $('#fx-eurc').textContent = fx.eurc_balance.toFixed(0) + ' (' + fmtUsd(eurcUsd) + ')';
    $('#fx-usdc').textContent = fmtUsd(fx.usdc_balance);
    $('#fx-exposure').textContent = expPct;
    const feePct = fx.maker_fee_pct != null ? (fx.maker_fee_pct * 100).toFixed(1) + '%' : '--';
    $('#fx-fee').textContent = feePct;

    updateCurrencyChart(fx, data.holdings);
  } else {
    $('#cash-total').textContent = fmtCurr(data.cash);
    $('#cash-usdc-line').textContent = 'USDC: ' + fmtUsd(data.cash);
    $('#cash-eurc-line').textContent = '';
    $('#usd-exposure').textContent = 'N/A';
  }

  $('#total-value').textContent = fmtCurr(data.total_value);
  $('#holdings-value').textContent = fmtCurr(data.holdings);
  $('#floor-value').textContent = fmtCurr(data.capital_floor);

  // Split holdings into USD vs EUR denominated positions
  const positions = data.positions || [];
  let holdingsUsd = 0, holdingsEurUsd = 0;
  for (const p of positions) {
    if (p.product_id && p.product_id.endsWith('-EURC')) {
      holdingsEurUsd += (p.value_usd || 0);
    } else {
      holdingsUsd += (p.value_usd || 0);
    }
  }
  $('#holdings-usd-line').textContent = 'USD: ' + fmtUsd(holdingsUsd);
  if (latestEurUsdRate > 0) {
    const holdingsEur = holdingsEurUsd / latestEurUsdRate;
    $('#holdings-eur-line').textContent = 'EUR: ' + holdingsEur.toFixed(0) + ' (\u2248' + fmtUsd(holdingsEurUsd) + ')';
  } else {
    $('#holdings-eur-line').textContent = holdingsEurUsd > 0 ? 'EUR: \u2248' + fmtUsd(holdingsEurUsd) : '';
  }

  const pnlEl = $('#pnl-value');
  pnlEl.textContent = fmtCurr(data.unrealized_pnl);
  pnlEl.className = 'card-value ' + (data.unrealized_pnl >= 0 ? 'positive' : 'negative');

  setTrend('trend-total-value', toDisplay(data.total_value), 'portfolio', { format: 'curr' });
  setTrend('trend-holdings', toDisplay(data.holdings), 'holdings', { format: 'curr' });
  setTrend('trend-floor', toDisplay(data.capital_floor), 'floor', { format: 'curr' });
  setTrend('trend-pnl', toDisplay(data.unrealized_pnl), 'pnl', { format: 'curr' });

  updatePositions(data.positions || []);
}

// ---- Currency doughnut ----

function updateCurrencyChart(fx, holdingsUsd) {
  const eurcUsd = fx.eurc_balance * fx.eur_usd_rate;
  const usdcCash = fx.usdc_balance;
  const holdings = holdingsUsd || 0;

  const values = [eurcUsd, usdcCash, holdings];
  const labels = [
    'EURC (EUR) — parked',
    'USDC — idle cash',
    'Holdings (USD) — in trades',
  ];

  if (currencyChart) {
    currencyChart.data.datasets[0].data = values;
    currencyChart.update('none');
  } else {
    currencyChart = new Chart($('#currency-chart'), {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: ['#3b82f6', '#22c55e', '#eab308'],
          borderColor: ['#2563eb', '#16a34a', '#ca8a04'],
          borderWidth: 2,
          hoverOffset: 6,
        }]
      },
      options: {
        responsive: true,
        cutout: '60%',
        plugins: {
          legend: {
            position: 'right',
            labels: { color: '#8b8fa3', padding: 10, font: { size: 11 }, boxWidth: 12 }
          },
          tooltip: {
            callbacks: {
              label: ctx => {
                const val = ctx.parsed;
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                const pctVal = total > 0 ? (val / total * 100).toFixed(1) : '0';
                return ctx.label + ': ' + fmtUsd(val) + ' (' + pctVal + '%)';
              }
            }
          }
        }
      }
    });
  }
}

// ---- Risk / Drawdown ----

async function updateRisk() {
  const data = await fetchJSON('/risk');
  if (!data) return;

  const dd = data.drawdown_from_peak_pct;
  const ddEl = $('#drawdown-value');
  ddEl.textContent = dd > 0 ? '-' + pct(dd) : pct(dd);
  if (dd > 10) ddEl.className = 'card-value negative';
  else if (dd > 0) ddEl.className = 'card-value';
  else ddEl.className = 'card-value positive';

  setTrend('trend-drawdown', dd, 'drawdown', { format: 'pct', invert: true, threshold: 0.01 });
}

// ---- Positions table ----

function updatePositions(positions) {
  const tbody = $('#positions-table tbody');
  const frag = document.createDocumentFragment();
  if (!positions.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="11" style="color:var(--text-muted);text-align:center;">No open positions</td>';
    frag.appendChild(tr);
    tbody.replaceChildren(frag);
    return;
  }
  for (const p of positions) {
    const pnlClass = p.unrealized_pnl >= 0 ? 'positive' : 'negative';
    const strat = (p.strategy === 'momentum') ? ' <span class="badge badge-on" style="font-size:0.6rem;padding:0.1rem 0.4rem;">MOM</span>' : '';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong class="coin-link" data-pid="${p.product_id}">${p.product_id}</strong>${strat}</td>
      <td class="mono">${fmtPrice(p.entry_price)}</td>
      <td class="mono">${fmtPrice(p.current_price)}</td>
      <td class="mono">${fmtSize(p.size)}</td>
      <td class="mono">${fmtUsd(p.value_usd)}</td>
      <td class="mono">${p.fee != null ? fmtUsd(p.fee) : '--'}</td>
      <td class="mono ${pnlClass}">${fmtUsd(p.unrealized_pnl)}</td>
      <td class="${pnlClass}">${p.unrealized_pnl_pct >= 0 ? '+' : ''}${p.unrealized_pnl_pct.toFixed(1)}%</td>
      <td class="mono">${fmtPrice(p.stop_loss)}</td>
      <td class="mono">${fmtPrice(p.take_profit)}</td>
      <td class="pos-actions">
        <button class="btn-reset-sl" data-pid="${p.product_id}" title="Reset stop to current price −1%">Reset SL</button>
        <button class="btn-sell-pos" data-pid="${p.product_id}">Sell</button>
      </td>
    `;
    frag.appendChild(tr);
  }
  tbody.replaceChildren(frag);

  tbody.querySelectorAll('.btn-reset-sl').forEach(btn => {
    btn.addEventListener('click', async () => {
      const pid = btn.dataset.pid;
      btn.disabled = true;
      btn.textContent = '...';
      try {
        const res = await fetch(API + '/reset-stop/' + encodeURIComponent(pid), { method: 'POST' });
        const data = await res.json();
        if (data.ok) {
          btn.textContent = fmtPrice(data.stop_loss).replace('$', '');
          btn.className = 'btn-reset-sl done';
          setTimeout(refresh, 1000);
        } else {
          btn.textContent = 'Fail';
          setTimeout(() => { btn.textContent = 'Reset SL'; btn.disabled = false; btn.className = 'btn-reset-sl'; }, 2000);
        }
      } catch (e) {
        console.error('Reset SL error:', e);
        btn.textContent = 'Err';
        setTimeout(() => { btn.textContent = 'Reset SL'; btn.disabled = false; btn.className = 'btn-reset-sl'; }, 2000);
      }
    });
  });

  tbody.querySelectorAll('.btn-sell-pos').forEach(btn => {
    btn.addEventListener('click', async () => {
      const pid = btn.dataset.pid;
      if (!confirm(`Sell ${pid} at market price?`)) return;
      btn.disabled = true;
      btn.textContent = 'Selling...';
      try {
        const res = await fetch(API + '/sell/' + encodeURIComponent(pid), { method: 'POST' });
        const data = await res.json();
        if (data.closed) {
          btn.textContent = 'Sold!';
          btn.className = 'btn-sell-pos sold';
          setTimeout(refresh, 1000);
        } else {
          btn.textContent = 'Failed';
          setTimeout(() => { btn.textContent = 'Sell'; btn.disabled = false; }, 2000);
        }
      } catch (e) {
        console.error('Sell error:', e);
        btn.textContent = 'Error';
        setTimeout(() => { btn.textContent = 'Sell'; btn.disabled = false; }, 2000);
      }
    });
  });
}

// ---- Trades table ----

async function updateTrades() {
  const trades = await fetchJSON('/trades?limit=20');
  if (!trades) return;

  const tbody = $('#trades-table tbody');
  const frag = document.createDocumentFragment();
  if (!trades.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="9" style="color:var(--text-muted);text-align:center;">No trades yet</td>';
    frag.appendChild(tr);
    tbody.replaceChildren(frag);
    return;
  }
  const blockedIds = new Set(_cachedBlocklist.map(b => b.product_id));

  for (const t of trades) {
    const sideClass = t.side === 'BUY' ? 'buy' : 'sell';
    const time = t.created_at ? new Date(t.created_at).toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}) : '';
    const strat = (t.strategy === 'momentum') ? '<span class="badge badge-on" style="font-size:0.6rem;padding:0.1rem 0.4rem;">MOM</span>' : '';
    let pnlHtml = '--';
    if (t.pnl != null) {
      const cls = t.pnl >= 0 ? 'buy' : 'sell';
      const sign = t.pnl >= 0 ? '+' : '';
      pnlHtml = `<span class="${cls}">${sign}${fmtUsd(t.pnl)}</span>`;
    }
    const isBlocked = blockedIds.has(t.product_id);
    const blockBtnHtml = isBlocked
      ? `<button class="btn-block-coin blocked" data-pid="${t.product_id}" disabled>Blocked</button>`
      : `<button class="btn-block-coin" data-pid="${t.product_id}">Block</button>`;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${time}</td>
      <td><strong class="coin-link" data-pid="${t.product_id}">${t.product_id}</strong> ${strat}</td>
      <td class="${sideClass}">${t.side}</td>
      <td class="mono">${fmtPrice(t.price)}</td>
      <td class="mono">${fmtUsd(t.value || 0)}</td>
      <td class="mono">${t.fee != null ? fmtUsd(t.fee) : '--'}</td>
      <td class="mono">${pnlHtml}</td>
      <td>${t.status}</td>
      <td>${blockBtnHtml}</td>
    `;
    frag.appendChild(tr);
  }
  tbody.replaceChildren(frag);

  tbody.querySelectorAll('.btn-block-coin:not(.blocked)').forEach(btn => {
    btn.addEventListener('click', async () => {
      const pid = btn.dataset.pid;
      btn.disabled = true;
      btn.textContent = '...';
      try {
        await fetch(API + '/block/' + encodeURIComponent(pid), { method: 'POST' });
        btn.textContent = 'Blocked';
        btn.className = 'btn-block-coin blocked';
        updateBlocklist();
      } catch (e) {
        console.error('Block error:', e);
        btn.textContent = 'Err';
        setTimeout(() => { btn.textContent = 'Block'; btn.disabled = false; btn.className = 'btn-block-coin'; }, 2000);
      }
    });
  });
}

// ---- Blocklist ----

async function updateBlocklist() {
  const list = await fetchJSON('/blocklist');
  _cachedBlocklist = list || [];
  const section = $('#blocklist-section');
  if (!section) return;

  const tbody = $('#blocklist-table tbody');
  const frag = document.createDocumentFragment();
  if (!_cachedBlocklist.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="4" style="color:var(--text-muted);text-align:center;">No blocked coins</td>';
    frag.appendChild(tr);
  } else {
    for (const b of _cachedBlocklist) {
      const when = b.blocked_at ? new Date(b.blocked_at).toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}) : '';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${b.product_id}</strong></td>
        <td>${b.reason || 'manual'}</td>
        <td class="mono">${when}</td>
        <td><button class="btn-unblock-coin" data-pid="${b.product_id}">Unblock</button></td>
      `;
      frag.appendChild(tr);
    }
  }
  tbody.replaceChildren(frag);

  tbody.querySelectorAll('.btn-unblock-coin').forEach(btn => {
    btn.addEventListener('click', async () => {
      const pid = btn.dataset.pid;
      btn.disabled = true;
      btn.textContent = '...';
      try {
        await fetch(API + '/unblock/' + encodeURIComponent(pid), { method: 'POST' });
        updateBlocklist();
        updateTrades();
      } catch (e) {
        console.error('Unblock error:', e);
        btn.textContent = 'Err';
        setTimeout(() => { btn.textContent = 'Unblock'; btn.disabled = false; }, 2000);
      }
    });
  });
}

// ---- Equity + Drawdown charts ----

const chartColors = {
  line: '#5b7fff',
  lineFill: 'rgba(91,127,255,0.08)',
  red: '#ef4444',
  redFill: 'rgba(239,68,68,0.08)',
  grid: '#2a2d3a',
  tick: '#8b8fa3',
};

function linearRegression(values) {
  const n = values.length;
  if (n < 2) return values.slice();
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += values[i];
    sumXY += i * values[i];
    sumXX += i * i;
  }
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  return values.map((_, i) => intercept + slope * i);
}

function makeChartOpts(yOpts = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: 'index' },
    plugins: {
      legend: {
        display: true,
        labels: { color: chartColors.tick, boxWidth: 14, padding: 10, font: { size: 11 } }
      }
    },
    scales: {
      x: {
        display: true,
        ticks: { color: chartColors.tick, maxTicksLimit: 8, font: { size: 10 } },
        grid: { color: chartColors.grid }
      },
      y: {
        ticks: { color: chartColors.tick, font: { size: 10 } },
        grid: { color: chartColors.grid },
        ...yOpts,
      }
    }
  };
}

function timeLabelFormat(hours) {
  if (hours <= 8)    return { hour: '2-digit', minute: '2-digit' };
  if (hours <= 24)   return { hour: '2-digit', minute: '2-digit' };
  if (hours <= 168)  return { month: 'short', day: 'numeric', hour: '2-digit' };
  if (hours <= 720)  return { month: 'short', day: 'numeric' };
  return { year: 'numeric', month: 'short' };
}

// Wire up global time selector
document.querySelectorAll('#global-time-selector .time-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#global-time-selector .time-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedHours = parseInt(btn.dataset.hours, 10);
    refresh();
  });
});

// Wire up global currency toggle
document.querySelectorAll('#global-currency-toggle .currency-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#global-currency-toggle .currency-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    displayCurrency = btn.dataset.currency;
    refresh();
  });
});


async function updateCharts() {
  const hoursParam = selectedHours > 0 ? `hours=${selectedHours}` : 'hours=87600';
  const data = await fetchJSON(`/equity?${hoursParam}`);
  if (!data || !data.length) return;

  const fmt = timeLabelFormat(selectedHours || 87600);
  const labels = data.map(d => new Date(d.timestamp).toLocaleString('en-US', fmt));
  const rawValues = data.map(d => d.total_value);
  const dds = data.map(d => Math.round(-(d.drawdown_pct || 0) * 10000) / 100);

  const isEur = displayCurrency === 'EUR' && latestEurUsdRate > 0;
  const fxDiv = isEur ? latestEurUsdRate : 1;
  const values = rawValues.map(v => v / fxDiv);
  const equityLabel = isEur ? 'Equity (\u20AC)' : 'Equity ($)';
  const trendLabel = isEur ? 'Trend (\u20AC)' : 'Trend ($)';
  const currSymbol = isEur ? '\u20AC' : '$';

  const equityTrend = linearRegression(values);

  const yOpts = {
    ticks: {
      color: chartColors.tick,
      font: { size: 10 },
      callback: v => currSymbol + v.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
    }
  };

  if (equityChart) {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = values;
    equityChart.data.datasets[0].label = equityLabel;
    equityChart.data.datasets[1].data = equityTrend;
    equityChart.data.datasets[1].label = trendLabel;
    equityChart.options.scales.y.ticks = yOpts.ticks;
    equityChart.update('none');
  } else {
    equityChart = new Chart($('#equity-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: equityLabel,
          data: values,
          borderColor: chartColors.line,
          borderWidth: 2,
          fill: { target: 'origin', above: chartColors.lineFill },
          pointRadius: 0,
          tension: 0.3,
        }, {
          label: trendLabel,
          data: equityTrend,
          borderColor: '#f59e0b',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
        }]
      },
      options: makeChartOpts(yOpts)
    });
  }

  const ddTrend = linearRegression(dds);

  if (drawdownChart) {
    drawdownChart.data.labels = labels;
    drawdownChart.data.datasets[0].data = dds;
    drawdownChart.data.datasets[1].data = ddTrend;
    drawdownChart.update('none');
  } else {
    drawdownChart = new Chart($('#drawdown-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Drawdown',
          data: dds,
          borderColor: chartColors.red,
          backgroundColor: chartColors.redFill,
          borderWidth: 2,
          fill: 'origin',
          pointRadius: 0,
          tension: 0.3,
        }, {
          label: 'Trend',
          data: ddTrend,
          borderColor: '#f59e0b',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
        }]
      },
      options: makeChartOpts({
        max: 0,
        ticks: { color: chartColors.tick, font: { size: 10 }, callback: v => v.toFixed(2) + '%' }
      })
    });
  }
}

// ---- P&L by Strategy chart (individual trades) ----

async function updateStrategyChart() {
  const data = await fetchJSON('/strategy-performance');
  if (!data) return;

  const STRATS = [
    { key: 'ml',       label: 'ML',       color: '#22c55e', ptColor: d => d.raw.pnl >= 0 ? '#22c55e' : '#ef4444' },
    { key: 'momentum', label: 'Momentum', color: '#f59e0b', ptColor: d => d.raw.pnl >= 0 ? '#f59e0b' : '#ef4444' },
    { key: 'external', label: 'External', color: '#8b5cf6', ptColor: d => d.raw.pnl >= 0 ? '#8b5cf6' : '#ef4444' },
  ];

  // Time filter
  const cutoff = selectedHours > 0
    ? new Date(Date.now() - selectedHours * 3600_000).toISOString()
    : null;

  const datasets = [];
  const sym = currSym();
  const fmtPnl = v => {
    const dv = toDisplay(v);
    return (dv >= 0 ? '+' + sym : '-' + sym) + Math.abs(dv).toFixed(2);
  };

  for (const s of STRATS) {
    let trades = (data[s.key] || []).filter(t => t.timestamp);
    if (cutoff) trades = trades.filter(t => t.timestamp >= cutoff);
    if (trades.length === 0) continue;

    const points = trades.map(t => ({
      x: new Date(t.timestamp),
      y: toDisplay(t.pnl),
      raw: t,
    }));

    datasets.push({
      label: s.label,
      data: points,
      borderColor: s.color,
      borderWidth: 1.5,
      pointRadius: 4,
      pointHoverRadius: 7,
      pointBackgroundColor: points.map(s.ptColor),
      pointBorderColor: points.map(s.ptColor),
      fill: false,
      tension: 0.15,
      spanGaps: true,
    });
  }

  if (datasets.length === 0) return;

  // Zero line
  const allTimes = datasets.flatMap(ds => ds.data.map(p => p.x.getTime()));
  const tMin = new Date(Math.min(...allTimes));
  const tMax = new Date(Math.max(...allTimes));
  datasets.push({
    label: '_zero',
    data: [{ x: tMin, y: 0 }, { x: tMax, y: 0 }],
    borderColor: 'rgba(139,143,163,0.3)',
    borderWidth: 1,
    borderDash: [4, 4],
    pointRadius: 0,
    fill: false,
  });

  const subtitleParts = STRATS.map(s => {
    const tot = data.totals ? data.totals[s.key] : 0;
    return `${s.label}: ${fmtPnl(tot)}`;
  });
  subtitleParts.push(`Combined: ${fmtPnl(data.totals ? data.totals.combined : 0)}`);

  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: true },
    scales: {
      x: {
        type: 'time',
        time: { tooltipFormat: 'MMM d, HH:mm' },
        grid: { color: 'rgba(139,143,163,0.08)' },
        ticks: { color: '#8b8fa3', font: { size: 10 }, maxTicksLimit: 12 },
      },
      y: {
        grid: { color: 'rgba(139,143,163,0.08)' },
        ticks: {
          color: '#8b8fa3',
          font: { size: 10 },
          callback: v => (v >= 0 ? '+' : '-') + sym + Math.abs(v).toFixed(2),
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        labels: {
          color: chartColors.tick,
          boxWidth: 14,
          padding: 10,
          font: { size: 11 },
          filter: item => !item.text.startsWith('_'),
        },
      },
      subtitle: {
        display: true,
        text: subtitleParts.join('   '),
        color: '#8b8fa3',
        font: { size: 11 },
        padding: { bottom: 8 },
      },
      tooltip: {
        callbacks: {
          title: ctx => {
            const p = ctx[0];
            return p.raw.raw ? p.raw.raw.product_id : '';
          },
          label: ctx => {
            const v = ctx.raw.y;
            return ` ${ctx.dataset.label}: ${fmtPnl(v)}`;
          },
        },
      },
    },
  };

  if (strategyChart) {
    strategyChart.data.datasets = datasets;
    strategyChart.options.plugins.subtitle = opts.plugins.subtitle;
    strategyChart.update('none');
  } else {
    strategyChart = new Chart($('#strategy-chart'), {
      type: 'line',
      data: { datasets },
      options: opts,
    });
  }

  // Strategy stats below chart
  const stats = data.stats || {};
  const statsEl = $('#strat-stats');
  if (statsEl) {
    let html = '';
    for (const s of STRATS) {
      const st = stats[s.key];
      if (!st || st.trades === 0) continue;
      const wr = st.win_rate != null ? (st.win_rate * 100).toFixed(0) + '%' : '--';
      const wrCls = st.win_rate >= 0.5 ? 'positive' : 'negative';
      const total = data.totals ? data.totals[s.key] : 0;
      html += `<div class="strat-stat-row">` +
        `<span class="strat-stat-label" style="color:${s.color}">${s.label}</span>` +
        `<span class="strat-stat-detail">${st.trades} trades</span>` +
        `<span class="strat-stat-detail">${st.wins}W / ${st.trades - st.wins}L</span>` +
        `<span class="strat-stat-detail ${wrCls}">${wr} win</span>` +
        `<span class="strat-stat-detail">${fmtPnl(total)}</span>` +
        `</div>`;
    }
    statsEl.innerHTML = html || '<span style="color:var(--text-muted);font-size:0.75rem;">No closed trades yet</span>';
  }
}

// ---- Performance Attribution chart ----

async function updatePerformanceChart() {
  const perfHoursParam = selectedHours > 0 ? `hours=${selectedHours}` : 'hours=87600';
  const data = await fetchJSON(`/equity?${perfHoursParam}`);
  if (!data || !data.length) return;

  const withFx = data.filter(d => d.eur_usd_rate != null && d.eur_usd_rate > 0);
  if (withFx.length < 2) return;

  const v0 = withFx[0].total_value;
  const r0 = withFx[0].eur_usd_rate;
  if (!v0 || v0 <= 0 || !r0 || r0 <= 0) return;

  const fmt = timeLabelFormat(selectedHours || 87600);
  const labels = withFx.map(d => new Date(d.timestamp).toLocaleString('en-US', fmt));

  // Trading only: USD return (FX-neutral)
  const tradingPct = withFx.map(d => ((d.total_value / v0) - 1) * 100);
  // FX effect: how EUR/USD movement affects EUR value
  const fxPct = withFx.map(d => ((r0 / d.eur_usd_rate) - 1) * 100);
  // Total EUR return
  const totalEurPct = withFx.map(d => (((d.total_value / d.eur_usd_rate) / (v0 / r0)) - 1) * 100);

  const zeroLine = withFx.map(() => 0);

  if (performanceChart) {
    performanceChart.data.labels = labels;
    performanceChart.data.datasets[0].data = totalEurPct;
    performanceChart.data.datasets[1].data = tradingPct;
    performanceChart.data.datasets[2].data = fxPct;
    performanceChart.data.datasets[3].data = zeroLine;
    performanceChart.update('none');
  } else {
    performanceChart = new Chart($('#performance-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Total EUR Return',
          data: totalEurPct,
          borderColor: '#5b7fff',
          borderWidth: 2,
          fill: false,
          pointRadius: 0,
          tension: 0.3,
        }, {
          label: 'Trading Only (FX-neutral)',
          data: tradingPct,
          borderColor: '#22c55e',
          borderWidth: 2,
          fill: false,
          pointRadius: 0,
          tension: 0.3,
        }, {
          label: 'FX Effect',
          data: fxPct,
          borderColor: '#f59e0b',
          borderWidth: 1.5,
          borderDash: [6, 4],
          fill: false,
          pointRadius: 0,
          tension: 0.3,
        }, {
          label: 'Zero',
          data: zeroLine,
          borderColor: 'rgba(139,143,163,0.3)',
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        }]
      },
      options: makeChartOpts({
        ticks: {
          color: '#8b8fa3',
          font: { size: 10 },
          callback: v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%',
        },
      })
    });
  }
}

// ---- Crypto Confidence (benchmarks) chart ----

const benchColors = {
  BTC:  '#f7931a',
  ETH:  '#627eea',
  DOGE: '#c3a634',
  SOL:  '#9945ff',
};
const benchHidden = {};

async function updateBenchmarkChart() {
  const benchHours = selectedHours > 0 ? selectedHours : 87600;
  const [benchData, eqData] = await Promise.all([
    fetchJSON(`/benchmarks?hours=${benchHours}`),
    fetchJSON(`/equity?hours=${benchHours}`),
  ]);
  if (!benchData) return;

  // Find the coin with the most data points for time labels
  const coins = Object.keys(benchColors);
  let longestKey = coins[0];
  for (const k of coins) {
    if ((benchData[k] || []).length > (benchData[longestKey] || []).length) longestKey = k;
  }
  const refSeries = benchData[longestKey] || [];
  if (refSeries.length < 2) return;

  const fmt = timeLabelFormat(benchHours);
  const labels = refSeries.map(d => new Date(d.timestamp * 1000).toLocaleString('en-US', fmt));

  // Save hidden state before rebuilding
  if (benchmarkChart) {
    benchmarkChart.data.datasets.forEach((ds, i) => {
      if (ds.label) benchHidden[ds.label] = !benchmarkChart.isDatasetVisible(i);
    });
  }

  const datasets = [];
  const visibleBenchReturns = [];

  for (const coin of coins) {
    const arr = benchData[coin] || [];
    if (arr.length < 2) continue;
    const base = arr[0].close;
    if (!base || base <= 0) continue;
    const pctReturns = arr.map(d => ((d.close / base) - 1) * 100);
    const isHidden = !!benchHidden[coin];
    datasets.push({
      label: coin,
      data: pctReturns,
      borderColor: benchColors[coin],
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
      hidden: isHidden,
    });
    if (!isHidden) visibleBenchReturns.push(pctReturns);
  }

  // Market trend: average of visible benchmark coins, then linear regression
  if (visibleBenchReturns.length > 0) {
    const len = Math.min(...visibleBenchReturns.map(a => a.length));
    const avg = [];
    for (let i = 0; i < len; i++) {
      let sum = 0;
      for (const r of visibleBenchReturns) sum += r[i];
      avg.push(sum / visibleBenchReturns.length);
    }
    const marketTrend = linearRegression(avg);
    datasets.push({
      label: 'Market Trend',
      data: marketTrend,
      borderColor: 'rgba(139,143,163,0.7)',
      borderWidth: 1.5,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
      tension: 0,
    });
  }

  // Trading return (from equity data)
  let tradingPct = null;
  if (eqData && eqData.length >= 2) {
    const v0 = eqData[0].total_value;
    if (v0 && v0 > 0) {
      tradingPct = eqData.map(d => ((d.total_value / v0) - 1) * 100);
      datasets.push({
        label: 'My Trading',
        data: tradingPct,
        borderColor: '#22c55e',
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        hidden: !!benchHidden['My Trading'],
      });
      if (!benchHidden['My Trading']) {
        datasets.push({
          label: 'My Trend',
          data: linearRegression(tradingPct),
          borderColor: 'rgba(34,197,94,0.5)',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
        });
      }
    }
  }

  // Zero baseline (hidden from legend)
  datasets.push({
    label: '_zero',
    data: refSeries.map(() => 0),
    borderColor: 'rgba(139,143,163,0.3)',
    borderWidth: 1,
    borderDash: [4, 4],
    pointRadius: 0,
    fill: false,
  });

  const benchOpts = makeChartOpts({
    ticks: {
      color: '#8b8fa3',
      font: { size: 10 },
      callback: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%',
    },
  });
  benchOpts.plugins.legend = {
    display: true,
    labels: {
      color: chartColors.tick,
      boxWidth: 14,
      padding: 10,
      font: { size: 11 },
      filter: item => !item.text.startsWith('_') && !item.text.endsWith('Trend'),
      usePointStyle: true,
      pointStyle: 'circle',
    },
    onClick: (e, legendItem, legend) => {
      const idx = legendItem.datasetIndex;
      const chart = legend.chart;
      const ds = chart.data.datasets[idx];
      const nowVisible = chart.isDatasetVisible(idx);
      chart.setDatasetVisibility(idx, !nowVisible);
      if (ds && ds.label) benchHidden[ds.label] = nowVisible;
      updateBenchmarkChart();
    },
  };

  if (benchmarkChart) {
    benchmarkChart.data.labels = labels;
    benchmarkChart.data.datasets = datasets;
    benchmarkChart.update('none');
  } else {
    benchmarkChart = new Chart($('#benchmark-chart'), {
      type: 'line',
      data: { labels, datasets },
      options: benchOpts,
    });
  }
}

// ---- FX Evolution chart ----

async function updateFxEvolutionChart() {
  const fxHoursParam = selectedHours > 0 ? `hours=${selectedHours}` : 'hours=87600';
  const data = await fetchJSON(`/equity?${fxHoursParam}`);
  if (!data || !data.length) return;

  const fxPoints = data.filter(d => d.eur_usd_rate != null);
  if (!fxPoints.length) {
    if (!fxEvolutionChart) {
      const ctx = $('#fx-evolution-chart');
      if (ctx) {
        fxEvolutionChart = new Chart(ctx, {
          type: 'line',
          data: { labels: ['--'], datasets: [{ label: 'EUR/USD', data: [0], borderColor: chartColors.line, pointRadius: 0 }] },
          options: makeChartOpts()
        });
      }
    }
    return;
  }

  const fmt = timeLabelFormat(selectedHours || 87600);
  const labels = fxPoints.map(d => new Date(d.timestamp).toLocaleString('en-US', fmt));
  const rates = fxPoints.map(d => d.eur_usd_rate);
  const trend = linearRegression(rates);

  if (fxEvolutionChart) {
    fxEvolutionChart.data.labels = labels;
    fxEvolutionChart.data.datasets[0].data = rates;
    fxEvolutionChart.data.datasets[1].data = trend;
    fxEvolutionChart.update('none');
  } else {
    fxEvolutionChart = new Chart($('#fx-evolution-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'EUR/USD',
          data: rates,
          borderColor: '#3b82f6',
          borderWidth: 2,
          fill: { target: 'origin', above: 'rgba(59,130,246,0.08)' },
          pointRadius: 0,
          tension: 0.3,
        }, {
          label: 'Trend',
          data: trend,
          borderColor: '#f59e0b',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
        }]
      },
      options: makeChartOpts({
        ticks: {
          color: chartColors.tick,
          font: { size: 10 },
          callback: v => v.toFixed(4),
        }
      })
    });
  }
}

// ---- Momentum ----

async function updateMomentum() {
  const data = await fetchJSON('/momentum/status');
  if (!data) return;

  const badge = $('#momentum-status');
  const btn = $('#momentum-toggle');
  const scanBtn = $('#momentum-scan-now');
  if (data.enabled || data.scanner_running) {
    badge.textContent = 'ACTIVE';
    badge.className = 'badge badge-on';
    btn.textContent = 'Stop';
    btn.className = 'btn-momentum active';
    if (scanBtn) scanBtn.style.display = 'inline-block';
  } else {
    badge.textContent = 'INACTIVE';
    badge.className = 'badge badge-off';
    btn.textContent = 'Start';
    btn.className = 'btn-momentum';
    if (scanBtn) scanBtn.style.display = 'none';
  }

  if (data.last_scan_time) {
    const d = new Date(data.last_scan_time);
    $('#mom-last-scan').textContent = d.toLocaleTimeString();
  } else {
    $('#mom-last-scan').textContent = '--';
  }
  $('#mom-scanned').textContent = data.coins_scanned || 0;
  $('#mom-candidates').textContent = data.candidates_found || 0;
  $('#mom-positions').textContent = data.open_positions || 0;
  $('#mom-trades').textContent = data.total_trades || 0;

  // Confidence bar — per coin
  const conf = data.confidence || {};
  const confBar = $('#confidence-bar');
  if (confBar) {
    const parts = [];
    const coins = conf.coins || {};
    for (const [pid, c] of Object.entries(coins)) {
      if (!c || c.total === 0) continue;
      const pct = (c.accuracy * 100).toFixed(0);
      const cls = c.accuracy >= 0.7 ? 'confidence-high'
                : c.accuracy >= 0.5 ? 'confidence-mid' : 'confidence-low';
      const label = pid.replace('-USDC', '');
      parts.push(
        `<div class="confidence-item">` +
        `<span class="confidence-label">${label}</span>` +
        `<span class="confidence-pct ${cls}">${pct}%</span>` +
        `<span class="confidence-detail">(${c.correct}/${c.total})</span>` +
        `</div>`
      );
    }
    const ov = conf.overall;
    if (ov && ov.total > 0) {
      const ovPct = (ov.accuracy * 100).toFixed(0);
      const ovCls = ov.accuracy >= 0.7 ? 'confidence-high'
                  : ov.accuracy >= 0.5 ? 'confidence-mid' : 'confidence-low';
      parts.push(
        `<div class="confidence-item confidence-overall">` +
        `<span class="confidence-label">Overall</span>` +
        `<span class="confidence-pct ${ovCls}">${ovPct}%</span>` +
        `<span class="confidence-detail">(${ov.correct}/${ov.total})</span>` +
        `</div>`
      );
    }
    confBar.innerHTML = parts.length
      ? parts.join('')
      : '<span class="confidence-detail">No decision history yet</span>';
  }

  // Confidence evolution chart (learning curve)
  const series = (conf.series || []);
  const confChartWrap = document.querySelector('.confidence-chart-wrap');
  if (series.length >= 2) {
    if (confChartWrap) confChartWrap.style.display = '';
    const seriesLabels = series.map(s => {
      const d = new Date(s.timestamp * 1000);
      return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    });
    const accuracyData = series.map(s => s.accuracy * 100);
    const confTrend = linearRegression(accuracyData);

    const pointColors = series.map(s =>
      s.outcome === 'correct' ? 'rgba(34,197,94,0.9)' : 'rgba(239,68,68,0.9)'
    );

    if (confidenceChart) {
      confidenceChart.data.labels = seriesLabels;
      confidenceChart.data.datasets[0].data = accuracyData;
      confidenceChart.data.datasets[0].pointBackgroundColor = pointColors;
      confidenceChart.data.datasets[1].data = confTrend;
      confidenceChart.update('none');
    } else {
      const ctx = $('#confidence-chart');
      if (ctx) {
        confidenceChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: seriesLabels,
            datasets: [{
              label: 'Rolling Accuracy',
              data: accuracyData,
              borderColor: '#5b7fff',
              borderWidth: 2,
              fill: { target: 'origin', above: 'rgba(91,127,255,0.08)' },
              pointRadius: 3,
              pointBackgroundColor: pointColors,
              pointBorderWidth: 0,
              tension: 0.3,
            }, {
              label: 'Trend',
              data: confTrend,
              borderColor: '#f59e0b',
              borderWidth: 1.5,
              borderDash: [6, 4],
              pointRadius: 0,
              fill: false,
              tension: 0,
            }]
          },
          options: makeChartOpts({
            min: 0,
            max: 100,
            ticks: {
              color: '#8b8fa3',
              font: { size: 10 },
              callback: v => v + '%',
              stepSize: 25,
            },
          })
        });
      }
    }
  } else if (confChartWrap) {
    const pendingCount = (conf.recent || []).filter(r => !r.outcome || r.outcome === 'unknown').length;
    const totalRecent = (conf.recent || []).length;
    let msg = 'Waiting for decisions to be evaluated\u2026';
    if (totalRecent > 0) {
      msg = `${pendingCount} decision${pendingCount !== 1 ? 's' : ''} pending evaluation (~15 min)`;
    }
    confChartWrap.style.display = '';
    if (!confidenceChart) {
      const ctx = $('#confidence-chart');
      if (ctx) {
        confidenceChart = new Chart(ctx, {
          type: 'line',
          data: { labels: [''], datasets: [{ label: 'Rolling Accuracy', data: [null], borderColor: '#5b7fff', pointRadius: 0 }] },
          options: {
            ...makeChartOpts({ min: 0, max: 100, ticks: { color: '#8b8fa3', font: { size: 10 }, callback: v => v + '%', stepSize: 25 } }),
            plugins: {
              legend: { display: false },
              title: { display: true, text: msg, color: '#8b8fa3', font: { size: 12, weight: 'normal' }, padding: { top: 60 } }
            }
          }
        });
      }
    } else if (confidenceChart.options.plugins.title) {
      confidenceChart.options.plugins.title.text = msg;
      confidenceChart.update('none');
    }
  }

  // Show/hide Close All button
  const closeAllBtn = $('#momentum-close-all');
  const positions = data.positions || [];
  if (closeAllBtn) {
    closeAllBtn.style.display = positions.length > 0 ? 'inline-block' : 'none';
  }

  // Watchlist table
  const wl = data.watchlist || [];
  const wlBody = $('#momentum-watchlist tbody');
  const wlFrag = document.createDocumentFragment();
  if (!wl.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="10" style="color:var(--text-muted);text-align:center;">No candidates</td>';
    wlFrag.appendChild(tr);
  } else {
    for (const w of wl) {
      const changePct = (w.hourly_change_pct * 100).toFixed(1);
      const watchMin = Math.floor(w.watching_sec / 60);
      const watchSec = w.watching_sec % 60;
      const ttlMin = Math.floor(w.ttl_sec / 60);
      const ttlSec = w.ttl_sec % 60;

      // RSI coloring
      let rsiHtml = '--';
      if (w.rsi != null) {
        let rsiClass = '';
        if (w.rsi >= 70) rsiClass = 'rsi-overbought';
        else if (w.rsi <= 30) rsiClass = 'rsi-oversold';
        rsiHtml = `<span class="${rsiClass}">${w.rsi.toFixed(0)}</span>`;
      }

      // Range bar
      let rangeHtml = '--';
      if (w.range_position != null) {
        const rngPct = (w.range_position * 100).toFixed(0);
        let barClass = 'range-mid';
        if (w.range_position >= 0.80) barClass = 'range-high';
        else if (w.range_position <= 0.20) barClass = 'range-low';
        rangeHtml = `<div class="range-bar-wrap"><div class="range-bar ${barClass}" style="width:${rngPct}%"></div><span class="range-label">${rngPct}%</span></div>`;
      }

      // Trend
      let trendHtml = '--';
      if (w.short_trend != null) {
        const trendPct = (w.short_trend * 100).toFixed(1);
        const trendCls = w.short_trend >= 0 ? 'positive' : 'negative';
        const arrow = w.short_trend >= 0 ? '&#9650;' : '&#9660;';
        trendHtml = `<span class="${trendCls}">${arrow} ${w.short_trend >= 0 ? '+' : ''}${trendPct}%</span>`;
      }

      // Signal badge
      let signalHtml = '--';
      if (w.signal === 'pullback_watch') {
        const phase = w.pullback_phase || 'pullback';
        const phCls = phase === 'recovery' ? 'signal-recovery' : 'signal-pullback';
        const dropPct = w.pullback_drop_pct != null ? (w.pullback_drop_pct * 100).toFixed(1) : '?';
        const bouncePct = w.pullback_bounce_pct != null ? (w.pullback_bounce_pct * 100).toFixed(1) : '?';
        const progress = phase === 'pullback'
          ? `drop ${dropPct}% (need 3%)`
          : `bounce ${bouncePct}% (need 1.5%)`;
        signalHtml = `<span class="signal-badge ${phCls}" title="${progress}">${phase.toUpperCase()}</span>`;
      } else if (w.signal) {
        const sigCls = w.signal === 'enter' ? 'signal-enter' : w.signal === 'wait' ? 'signal-wait' : 'signal-skip';
        signalHtml = `<span class="signal-badge ${sigCls}" title="${w.reason || ''}">${w.signal.toUpperCase()}</span>`;
      }

      // Scale-in indicator
      const scaleTag = w.scale_in ? ' <span class="badge-scale-in">SI</span>' : '';

      // Progress column for pullback watches
      let progressHtml = '--';
      if (w.signal === 'pullback_watch') {
        const phase = w.pullback_phase || 'pullback';
        if (phase === 'pullback') {
          const drop = w.pullback_drop_pct != null ? (w.pullback_drop_pct * 100) : 0;
          const target = 3.0;
          const pctDone = Math.min(100, (drop / target) * 100).toFixed(0);
          progressHtml = `<div class="range-bar-wrap"><div class="range-bar pullback-bar" style="width:${pctDone}%"></div><span class="range-label">\u2193${drop.toFixed(1)}%</span></div>`;
        } else {
          const bounce = w.pullback_bounce_pct != null ? (w.pullback_bounce_pct * 100) : 0;
          const target = 1.5;
          const pctDone = Math.min(100, (bounce / target) * 100).toFixed(0);
          progressHtml = `<div class="range-bar-wrap"><div class="range-bar recovery-bar" style="width:${pctDone}%"></div><span class="range-label">\u2191${bounce.toFixed(1)}%</span></div>`;
        }
      }

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong class="coin-link" data-pid="${w.product_id}">${w.product_id}</strong>${scaleTag}</td>
        <td class="mono positive">+${changePct}%</td>
        <td class="mono">${w.current_price != null ? fmtPrice(w.current_price) : '--'}</td>
        <td class="mono">${rsiHtml}</td>
        <td>${rangeHtml}</td>
        <td class="mono">${trendHtml}</td>
        <td>${signalHtml}</td>
        <td>${progressHtml}</td>
        <td>${watchMin}m${watchSec.toString().padStart(2,'0')}s</td>
        <td>${ttlMin}m${ttlSec.toString().padStart(2,'0')}s</td>
      `;
      wlFrag.appendChild(tr);
    }
  }
  wlBody.replaceChildren(wlFrag);

  // Positions table
  const tbody = $('#momentum-table tbody');
  const momFrag = document.createDocumentFragment();
  if (!positions.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="6" style="color:var(--text-muted);text-align:center;">No momentum positions</td>';
    momFrag.appendChild(tr);
  }
  for (const p of positions) {
    let age = '--';
    if (p.opened_at) {
      const mins = Math.floor((Date.now() - new Date(p.opened_at).getTime()) / 60000);
      age = mins + 'min';
    }
    const tr = document.createElement('tr');
    const closeBtnHtml = `<button class="btn-close-one" data-pid="${p.product_id}">Close</button>`;
    tr.innerHTML = `
      <td><strong class="coin-link" data-pid="${p.product_id}">${p.product_id}</strong></td>
      <td class="mono">${fmtPrice(p.entry_price)}</td>
      <td class="mono">${fmtPrice(p.stop_loss)}</td>
      <td class="mono">${fmtPrice(p.take_profit)}</td>
      <td>${age}</td>
      <td>${closeBtnHtml}</td>
    `;
    momFrag.appendChild(tr);
  }
  tbody.replaceChildren(momFrag);

  // Bind per-position close buttons
  tbody.querySelectorAll('.btn-close-one').forEach(btn => {
    btn.addEventListener('click', async () => {
      const pid = btn.dataset.pid;
      btn.disabled = true;
      btn.textContent = 'Closing...';
      try {
        await fetch(API + '/momentum/close/' + encodeURIComponent(pid), { method: 'POST' });
        setTimeout(updateMomentum, 500);
      } catch (e) {
        console.error('Close error:', e);
        btn.textContent = 'Error';
      }
    });
  });

  // Recent decisions table
  const recent = (conf.recent || []);
  const dlBody = $('#decision-log-table tbody');
  if (dlBody) {
    const dlFrag = document.createDocumentFragment();
    if (!recent.length) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td colspan="7" style="color:var(--text-muted);text-align:center;">No decisions recorded yet</td>';
      dlFrag.appendChild(tr);
    } else {
      for (const r of recent) {
        const t = new Date(r.timestamp * 1000);
        const timeStr = t.toLocaleTimeString();
        const sigCls = r.decision === 'enter' ? 'signal-enter'
                     : r.decision === 'wait' ? 'signal-wait'
                     : r.decision === 'pullback_watch' ? 'signal-pullback'
                     : 'signal-skip';
        const sigLabel = r.decision === 'pullback_watch' ? 'PULLBACK' : r.decision.toUpperCase();
        const sigHtml = `<span class="signal-badge ${sigCls}">${sigLabel}</span>`;
        let outcomeHtml;
        if (r.outcome === 'correct') {
          outcomeHtml = '<span class="outcome-badge outcome-correct">Correct</span>';
        } else if (r.outcome === 'incorrect') {
          outcomeHtml = '<span class="outcome-badge outcome-incorrect">Wrong</span>';
        } else {
          outcomeHtml = '<span class="outcome-badge outcome-pending">Pending</span>';
        }
        let afterHtml = '--';
        if (r.price_change_pct != null) {
          const pctVal = (r.price_change_pct * 100).toFixed(1);
          const cls = r.price_change_pct >= 0 ? 'positive' : 'negative';
          afterHtml = `<span class="${cls}">${r.price_change_pct >= 0 ? '+' : ''}${pctVal}%</span>`;
        }
        const hourlyPct = (r.hourly_change_pct * 100).toFixed(1);
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono">${timeStr}</td>
          <td><strong class="coin-link" data-pid="${r.product_id}">${r.product_id}</strong></td>
          <td>${sigHtml}</td>
          <td class="mono">${fmtPrice(r.price)}</td>
          <td class="mono positive">+${hourlyPct}%</td>
          <td>${outcomeHtml}</td>
          <td class="mono">${afterHtml}</td>
        `;
        dlFrag.appendChild(tr);
      }
    }
    dlBody.replaceChildren(dlFrag);
  }
}

// ---- Fullscreen toggle ----

let fullscreenBackdrop = null;

function toggleFullscreen(panel) {
  const isFullscreen = panel.classList.contains('fullscreen');

  if (isFullscreen) {
    panel.classList.remove('fullscreen');
    if (fullscreenBackdrop) {
      fullscreenBackdrop.remove();
      fullscreenBackdrop = null;
    }
  } else {
    document.querySelectorAll('.chart-panel.fullscreen, .confidence-chart-wrap.fullscreen').forEach(p => {
      p.classList.remove('fullscreen');
    });
    if (fullscreenBackdrop) fullscreenBackdrop.remove();
    fullscreenBackdrop = document.createElement('div');
    fullscreenBackdrop.className = 'fullscreen-backdrop';
    fullscreenBackdrop.addEventListener('click', () => toggleFullscreen(panel));
    document.body.appendChild(fullscreenBackdrop);
    panel.classList.add('fullscreen');
  }

  const canvas = panel.querySelector('canvas');
  if (canvas) {
    const chartInstance = Chart.getChart(canvas);
    if (chartInstance) {
      setTimeout(() => chartInstance.resize(), 50);
    }
  }
}

document.querySelectorAll('.btn-fullscreen').forEach(btn => {
  btn.addEventListener('click', () => {
    const panel = btn.closest('.chart-panel') || btn.closest('.confidence-chart-wrap');
    if (panel) toggleFullscreen(panel);
  });
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const fs = document.querySelector('.chart-panel.fullscreen, .confidence-chart-wrap.fullscreen');
    if (fs) toggleFullscreen(fs);
  }
});

// ---- Main loop ----

let _refreshLock = false;
async function refresh() {
  if (_refreshLock) return;
  _refreshLock = true;

  try {
    await updateBlocklist();
    await Promise.all([updatePortfolio(), updateRisk(), updateTrades(), updateCharts(), updateStrategyChart(), updatePerformanceChart(), updateBenchmarkChart(), updateFxEvolutionChart(), updateMomentum()]);
    $('#status-badge').textContent = 'Connected';
    $('#status-badge').className = 'badge badge-ok';
  } catch (e) {
    $('#status-badge').textContent = 'Error';
    $('#status-badge').className = 'badge badge-err';
  }

  _refreshLock = false;
}

// Momentum buttons — use addEventListener for reliability
document.addEventListener('DOMContentLoaded', () => {
  const btn = $('#momentum-toggle');
  if (btn) {
    btn.addEventListener('click', async () => {
      const badge = $('#momentum-status');
      const isActive = badge.textContent.trim() === 'ACTIVE';
      const endpoint = isActive ? '/momentum/stop' : '/momentum/start';

      btn.disabled = true;
      btn.textContent = isActive ? 'Stopping...' : 'Starting...';

      try {
        const res = await fetch(API + endpoint, { method: 'POST' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        badge.textContent = isActive ? 'INACTIVE' : 'ACTIVE';
        badge.className = isActive ? 'badge badge-off' : 'badge badge-on';
        btn.textContent = isActive ? 'Start' : 'Stop';
        btn.className = isActive ? 'btn-momentum' : 'btn-momentum active';
      } catch (e) {
        console.error('Momentum toggle error:', e);
        btn.textContent = 'Error!';
        setTimeout(() => { btn.textContent = isActive ? 'Stop' : 'Start'; }, 2000);
      } finally {
        btn.disabled = false;
      }
    });
  }

  const closeAllBtn = $('#momentum-close-all');
  if (closeAllBtn) {
    closeAllBtn.addEventListener('click', async () => {
      closeAllBtn.disabled = true;
      closeAllBtn.textContent = 'Closing...';
      try {
        const res = await fetch(API + '/momentum/close-all', { method: 'POST' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        closeAllBtn.textContent = `Closed ${data.closed || 0}`;
        setTimeout(() => { closeAllBtn.style.display = 'none'; updateMomentum(); }, 1000);
      } catch (e) {
        console.error('Close all error:', e);
        closeAllBtn.textContent = 'Error!';
        setTimeout(() => { closeAllBtn.textContent = 'Close All'; }, 2000);
      } finally {
        closeAllBtn.disabled = false;
      }
    });
  }

  const scanNowBtn = $('#momentum-scan-now');
  if (scanNowBtn) {
    scanNowBtn.addEventListener('click', async () => {
      scanNowBtn.disabled = true;
      scanNowBtn.textContent = 'Scanning...';
      try {
        const res = await fetch(API + '/momentum/scan', { method: 'POST' });
        const data = await res.json();
        if (data.ok) {
          scanNowBtn.textContent = 'Triggered';
          setTimeout(() => { scanNowBtn.textContent = 'Scan Now'; scanNowBtn.disabled = false; updateMomentum(); }, 3000);
        } else {
          scanNowBtn.textContent = data.error || 'Failed';
          setTimeout(() => { scanNowBtn.textContent = 'Scan Now'; scanNowBtn.disabled = false; }, 2000);
        }
      } catch (e) {
        console.error('Scan error:', e);
        scanNowBtn.textContent = 'Error';
        setTimeout(() => { scanNowBtn.textContent = 'Scan Now'; scanNowBtn.disabled = false; }, 2000);
      }
    });
  }

  // Auto-sync status indicator
  const syncIndicator = $('#sync-indicator');
  async function updateSyncStatus() {
    try {
      const status = await fetchJSON('/sync-status');
      if (!status || !syncIndicator) return;
      if (!status.enabled) { syncIndicator.textContent = ''; return; }
      if (status.last_sync) {
        const ago = Math.round((Date.now() - new Date(status.last_sync + 'Z').getTime()) / 1000);
        const label = ago < 60 ? `${ago}s ago` : `${Math.round(ago / 60)}m ago`;
        const changes = status.last_result ? status.last_result.changes || 0 : 0;
        syncIndicator.textContent = changes > 0 ? `synced ${label} (${changes} fix)` : `synced ${label}`;
        syncIndicator.className = 'sync-indicator active';
      } else {
        syncIndicator.textContent = 'waiting...';
        syncIndicator.className = 'sync-indicator';
      }
    } catch (_) { /* ignore */ }
  }
  updateSyncStatus();
  setInterval(updateSyncStatus, 30000);

  // Sync button — reconcile DB with Coinbase
  const syncBtn = $('#btn-sync');
  if (syncBtn) {
    syncBtn.addEventListener('click', async () => {
      syncBtn.disabled = true;
      syncBtn.textContent = 'Checking...';
      try {
        const preview = await fetchJSON('/reconcile');
        if (!preview) throw new Error('Failed to fetch');

        const nPhantom = (preview.phantoms || []).length;
        const nOrphan = (preview.orphans || []).length;
        const nSize = (preview.size_updates || []).length;
        const nMatch = (preview.matched || []).length;

        if (nPhantom === 0 && nOrphan === 0 && nSize === 0) {
          syncBtn.textContent = 'In sync';
          syncBtn.className = 'btn-sync synced';
          setTimeout(() => { syncBtn.textContent = 'Sync'; syncBtn.className = 'btn-sync'; syncBtn.disabled = false; }, 2000);
          return;
        }

        const lines = [];
        if (nPhantom) {
          lines.push(`Close ${nPhantom} phantom position(s) (in DB but not on Coinbase):`);
          preview.phantoms.forEach(p => lines.push(`  - ${p.product_id} (size: ${p.db_size})`));
        }
        if (nOrphan) {
          lines.push(`Import ${nOrphan} orphan holding(s) (on Coinbase but not in DB):`);
          preview.orphans.forEach(o => lines.push(`  - ${o.product_id} (${o.coinbase_size} @ $${o.current_price})`));
        }
        if (nSize) {
          lines.push(`Update size for ${nSize} position(s):`);
          preview.size_updates.forEach(u => lines.push(`  - ${u.product_id} (${u.db_size} → ${u.coinbase_size})`));
        }
        lines.push(`\n${nMatch} position(s) already match.`);
        lines.push('\nApply these changes?');

        if (!confirm(lines.join('\n'))) {
          syncBtn.textContent = 'Sync';
          syncBtn.disabled = false;
          return;
        }

        syncBtn.textContent = 'Syncing...';
        const res = await fetch(API + '/reconcile', { method: 'POST' });
        const result = await res.json();
        const total = (result.phantoms_closed || []).length + (result.orphans_imported || []).length + (result.sizes_updated || []).length;
        syncBtn.textContent = `Fixed ${total}`;
        syncBtn.className = 'btn-sync synced';
        setTimeout(() => { refresh(); syncBtn.textContent = 'Sync'; syncBtn.className = 'btn-sync'; syncBtn.disabled = false; }, 1500);
      } catch (e) {
        console.error('Sync error:', e);
        syncBtn.textContent = 'Error';
        setTimeout(() => { syncBtn.textContent = 'Sync'; syncBtn.disabled = false; }, 2000);
      }
    });
  }

  // Sync trades button — import Coinbase order history
  const syncTradesBtn = $('#btn-sync-trades');
  if (syncTradesBtn) {
    syncTradesBtn.addEventListener('click', async () => {
      syncTradesBtn.disabled = true;
      syncTradesBtn.textContent = 'Checking...';
      try {
        const preview = await fetchJSON('/reconcile-trades');
        if (!preview) throw new Error('Failed to fetch');

        const nMissing = (preview.missing || []).length;

        if (nMissing === 0) {
          syncTradesBtn.textContent = 'In sync';
          syncTradesBtn.className = 'btn-sync synced';
          setTimeout(() => { syncTradesBtn.textContent = 'Sync'; syncTradesBtn.className = 'btn-sync'; syncTradesBtn.disabled = false; }, 2000);
          return;
        }

        const lines = [`Import ${nMissing} trade(s) from Coinbase:\n`];
        for (const m of preview.missing.slice(0, 15)) {
          const side = m.side === 'BUY' ? 'BUY ' : 'SELL';
          lines.push(`  ${side} ${m.product_id} — ${m.size} @ $${m.price.toFixed(2)} (fee $${m.fee.toFixed(4)})`);
        }
        if (nMissing > 15) lines.push(`  ... and ${nMissing - 15} more`);
        lines.push(`\n${preview.already_synced} trade(s) already in DB.`);
        lines.push('\nImport these trades?');

        if (!confirm(lines.join('\n'))) {
          syncTradesBtn.textContent = 'Sync';
          syncTradesBtn.disabled = false;
          return;
        }

        syncTradesBtn.textContent = 'Syncing...';
        const res = await fetch(API + '/reconcile-trades', { method: 'POST' });
        const result = await res.json();
        syncTradesBtn.textContent = `Imported ${result.imported || 0}`;
        syncTradesBtn.className = 'btn-sync synced';
        setTimeout(() => { refresh(); syncTradesBtn.textContent = 'Sync'; syncTradesBtn.className = 'btn-sync'; syncTradesBtn.disabled = false; }, 1500);
      } catch (e) {
        console.error('Trade sync error:', e);
        syncTradesBtn.textContent = 'Error';
        setTimeout(() => { syncTradesBtn.textContent = 'Sync'; syncTradesBtn.disabled = false; }, 2000);
      }
    });
  }
});

// ---- Coin chart modal ----

let coinModalChart = null;
let coinModalProductId = null;

function openCoinModal(productId) {
  coinModalProductId = productId;
  const modal = $('#coin-chart-modal');
  $('#coin-modal-title').textContent = productId.replace('-USDC', '') + ' / USDC';
  modal.style.display = 'block';
  document.body.style.overflow = 'hidden';

  modal.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
  modal.querySelector('.time-btn[data-hours="1"]').classList.add('active');
  loadCoinChart(productId, 1);
}

function closeCoinModal() {
  $('#coin-chart-modal').style.display = 'none';
  document.body.style.overflow = '';
  coinModalProductId = null;
  if (coinModalChart) { coinModalChart.destroy(); coinModalChart = null; }
}

async function loadCoinChart(productId, hours) {
  const data = await fetchJSON(`/coin-chart/${encodeURIComponent(productId)}?hours=${hours}`);
  if (!data || !data.candles || !data.candles.length) return;

  const dateFmt = hours >= 24
    ? { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }
    : { hour: '2-digit', minute: '2-digit' };
  const labels = data.candles.map(c => new Date(c.timestamp * 1000).toLocaleString('en-US', dateFmt));
  const closes = data.candles.map(c => c.close);

  const first = closes[0], last = closes[closes.length - 1];
  const isUp = last >= first;
  const lineColor = isUp ? '#22c55e' : '#ef4444';
  const fillColor = isUp ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)';

  const changePct = ((last - first) / first * 100).toFixed(2);
  const changeStr = (isUp ? '+' : '') + changePct + '%';

  if (coinModalChart) coinModalChart.destroy();

  coinModalChart = new Chart(document.getElementById('coin-modal-chart').getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: productId,
        data: closes,
        borderColor: lineColor,
        backgroundColor: fillColor,
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        subtitle: {
          display: true,
          text: `${fmtPrice(last)}  (${changeStr})`,
          color: lineColor,
          font: { size: 14, weight: '600' },
          padding: { bottom: 10 },
        },
        tooltip: {
          callbacks: {
            label: ctx => fmtPrice(ctx.parsed.y),
          }
        }
      },
      scales: {
        x: {
          ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12, color: '#8b8fa3', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#8b8fa3', font: { size: 10 }, callback: v => fmtPrice(v) },
        }
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  $('#coin-modal-close').addEventListener('click', closeCoinModal);
  document.querySelector('.coin-modal-backdrop').addEventListener('click', closeCoinModal);

  document.querySelectorAll('#coin-chart-modal .time-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#coin-chart-modal .time-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (coinModalProductId) loadCoinChart(coinModalProductId, parseInt(btn.dataset.hours));
    });
  });
});

document.addEventListener('click', e => {
  const link = e.target.closest('.coin-link');
  if (link) openCoinModal(link.dataset.pid);
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && $('#coin-chart-modal').style.display !== 'none') {
    closeCoinModal();
  }
});

refresh();
setInterval(refresh, REFRESH_MS);
