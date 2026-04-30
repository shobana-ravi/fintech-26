import React, { useState, useMemo, useEffect } from 'react';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area,
  BarChart, Bar, Cell
} from 'recharts';
import { 
  Settings, 
  TrendingUp, 
  ShieldCheck, 
  Activity, 
  RefreshCcw,
  LayoutDashboard,
  BrainCircuit,
  Calendar,
  Layers,
  Zap
} from 'lucide-react';
import { getHedgeRecommendation, getQuote, getHistory, placePaperOrder, getPaperOrders } from './services/hedgeApi';

const DTE = 30;
const CONTRACT_MULT = 100;

function computeMetricsFromHistory(points) {
  if (!points || points.length < 2) {
    return { return_1d: 0, return_5d: 0, realized_vol_20d: 0.2 };
  }
  const ordered = [...points].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  const last = ordered[ordered.length - 1];
  const r1 = typeof last.return_1d === 'number' && !Number.isNaN(last.return_1d) ? last.return_1d : 0;
  let r5 = 0;
  if (ordered.length >= 6) {
    const c0 = ordered[ordered.length - 1].close;
    const c5 = ordered[ordered.length - 6].close;
    if (c5 > 0) r5 = c0 / c5 - 1;
  }
  const returns = ordered
    .map((p) => p.return_1d)
    .filter((r) => typeof r === 'number' && !Number.isNaN(r));
  const tail = returns.slice(-20);
  let rv = 0.2;
  if (tail.length >= 5) {
    const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
    const variance =
      tail.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(tail.length - 1, 1);
    rv = Math.sqrt(variance) * Math.sqrt(252);
    if (!Number.isFinite(rv) || rv < 1e-6) rv = 0.2;
  }
  return { return_1d: r1, return_5d: r5, realized_vol_20d: Math.min(Math.max(rv, 0.01), 3) };
}

function buildModelFeaturePayload({
  ticker,
  spot,
  historyPoints,
  portfolioDeltaShares,
  currentHedgeShares,
}) {
  const { return_1d, return_5d, realized_vol_20d } = computeMetricsFromHistory(historyPoints);
  const strike = Math.round(spot);
  const T = DTE / 365;
  const dte_today = DTE;
  const sigma_next = Math.min(realized_vol_20d * 1.02 + 0.001, 2.5);
  const delta = 0.5;
  const gamma = 0.15;
  const theta = -6.0;
  const vega = 9.0;
  const callPrice = Math.max(spot * 0.02, 0.01);
  const portfolioGamma = gamma * CONTRACT_MULT;
  const portfolioTheta = theta * CONTRACT_MULT;
  const portfolioVega = vega * CONTRACT_MULT;

  return {
    ticker,
    portfolio_delta: portfolioDeltaShares,
    current_hedge_shares: currentHedgeShares,
    spot_today: spot,
    return_1d,
    return_5d,
    realized_vol_20d,
    sigma_next,
    strike,
    T,
    dte_today,
    call_price: callPrice,
    delta,
    gamma,
    theta,
    vega,
    portfolio_gamma: portfolioGamma,
    portfolio_theta: portfolioTheta,
    portfolio_vega: portfolioVega,
    option_pnl: 0,
  };
}

const HedgeDashboard = () => {
  // State for parameters
  const [ticker, setTicker] = useState('SPY');
  const [isUpdating, setIsUpdating] = useState(false);
  const [predictionError, setPredictionError] = useState('');
  const [predictionConfidence, setPredictionConfidence] = useState(null);
  const [tickerQuote, setTickerQuote] = useState(null);
  const [isQuoteLoading, setIsQuoteLoading] = useState(false);
  const [quoteError, setQuoteError] = useState('');
  const [historyData, setHistoryData] = useState([]);
  const [historyError, setHistoryError] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [executeMessage, setExecuteMessage] = useState('');
  const [executeError, setExecuteError] = useState('');
  const [paperOrders, setPaperOrders] = useState([]);
  const [recommendation, setRecommendation] = useState(null);
  const [portfolioDeltaInput, setPortfolioDeltaInput] = useState('48');
  const [currentHedgeInput, setCurrentHedgeInput] = useState('-20');

  // Ticker price data
  const tickerPrices = { 
    'SPY': 512.45, 
    'QQQ': 445.18,
    'DIA': 389.21, 
    'IWM': 204.12 
  };
  const currentPrice = tickerPrices[ticker];
  const csvBackedTickers = new Set(['DIA', 'IWM', 'QQQ', 'SPY']);
  const isCsvBackedTicker = csvBackedTickers.has(ticker);
  const displayedPrice = isCsvBackedTicker && tickerQuote?.price ? tickerQuote.price : currentPrice;
  const displayedChangePct =
    isCsvBackedTicker && tickerQuote?.change_pct !== undefined ? tickerQuote.change_pct : 0.15;
  const changeText = `${displayedChangePct >= 0 ? '+' : ''}${displayedChangePct.toFixed(2)}%`;
  const changeColor = displayedChangePct >= 0 ? 'text-green-600' : 'text-rose-600';
  
  const recomHedgePct = recommendation
    ? Math.round(Number(recommendation.predicted_hedge_ratio_bucket) * 100)
    : null;

  const fallbackChartData = useMemo(() => {
    return Array.from({ length: 15 }, (_, i) => ({
      date: `Day ${i + 1}`,
      close: displayedPrice,
      hedge_intensity: 0,
    }));
  }, [ticker, displayedPrice]);
  const chartData = historyData.length > 0 ? historyData : fallbackChartData;

  useEffect(() => {
    if (!isCsvBackedTicker) {
      setTickerQuote(null);
      setQuoteError('');
      return;
    }

    let isCancelled = false;
    const loadTickerQuote = async () => {
      setIsQuoteLoading(true);
      setQuoteError('');
      try {
        const quote = await getQuote(ticker);
        if (!isCancelled) {
          setTickerQuote(quote);
        }
      } catch (error) {
        if (!isCancelled) {
          setQuoteError(error.message || `Could not fetch ${ticker} quote`);
        }
      } finally {
        if (!isCancelled) {
          setIsQuoteLoading(false);
        }
      }
    };

    loadTickerQuote();
    return () => {
      isCancelled = true;
    };
  }, [isCsvBackedTicker, ticker]);

  useEffect(() => {
    let isCancelled = false;
    const loadPaperOrders = async () => {
      try {
        const data = await getPaperOrders(5);
        if (!isCancelled) {
          setPaperOrders(Array.isArray(data.orders) ? data.orders : []);
        }
      } catch (_error) {
        if (!isCancelled) {
          setPaperOrders([]);
        }
      }
    };
    loadPaperOrders();
    return () => {
      isCancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!isCsvBackedTicker) {
      setHistoryData([]);
      setHistoryError('');
      return;
    }

    let isCancelled = false;
    const loadHistory = async () => {
      setHistoryError('');
      try {
        const history = await getHistory(ticker, 15);
        if (!isCancelled) {
          setHistoryData(
            (history.points || []).map((point) => ({
              date: point.date,
              close: point.close,
              hedge_intensity: point.hedge_intensity,
            })),
          );
        }
      } catch (error) {
        if (!isCancelled) {
          setHistoryError(error.message || `Could not fetch ${ticker} history`);
          setHistoryData([]);
        }
      }
    };

    loadHistory();
    return () => {
      isCancelled = true;
    };
  }, [isCsvBackedTicker, ticker]);

  const handleUpdate = async () => {
    setIsUpdating(true);
    setPredictionError('');
    setRecommendation(null);

    try {
      let spot = displayedPrice;
      if (isCsvBackedTicker) {
        try {
          setIsQuoteLoading(true);
          setQuoteError('');
          const quote = await getQuote(ticker);
          setTickerQuote(quote);
          spot = quote.price;
        } catch (error) {
          setQuoteError(error.message || `Could not fetch ${ticker} quote`);
        } finally {
          setIsQuoteLoading(false);
        }
      }

      let historyPoints = [];
      if (isCsvBackedTicker) {
        try {
          const history = await getHistory(ticker, 60);
          historyPoints = history.points || [];
        } catch (_e) {
          historyPoints = [];
        }
      }

      const portfolioDelta = Number(portfolioDeltaInput);
      const currentHedge = Number(currentHedgeInput);
      if (!Number.isFinite(portfolioDelta)) {
        throw new Error('Portfolio delta must be a number (share-equivalent).');
      }
      if (!Number.isFinite(currentHedge)) {
        throw new Error('Current hedge shares must be a number.');
      }

      const payload = buildModelFeaturePayload({
        ticker,
        spot,
        historyPoints,
        portfolioDeltaShares: portfolioDelta,
        currentHedgeShares: currentHedge,
      });

      const prediction = await getHedgeRecommendation(payload);
      setRecommendation(prediction);
      setPredictionConfidence(prediction.prediction_confidence);
    } catch (error) {
      setPredictionError(error.message || 'Could not fetch recommendation');
      setPredictionConfidence(null);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleExecute = async () => {
    if (!recommendation) return;
    const qty = Math.round(Math.abs(recommendation.shares_to_trade));
    if (qty === 0) {
      setExecuteError('No shares to trade for this recommendation.');
      return;
    }
    setIsExecuting(true);
    setExecuteMessage('');
    setExecuteError('');
    try {
      const side = recommendation.shares_to_trade >= 0 ? 'buy' : 'sell';
      const order = await placePaperOrder({
        ticker,
        side,
        quantity: qty,
        hedge_percent: recomHedgePct ?? 0,
        price: displayedPrice,
      });
      setExecuteMessage(
        `Paper order filled: ${order.side.toUpperCase()} ${order.quantity} ${order.ticker} @ $${Number(order.filled_price).toFixed(2)}`,
      );
      const ordersResponse = await getPaperOrders(5);
      setPaperOrders(Array.isArray(ordersResponse.orders) ? ordersResponse.orders : []);
    } catch (error) {
      setExecuteError(error.message || 'Paper trade failed');
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-12">
      {/* Navigation */}
      <nav className="bg-white border-b border-slate-200 px-8 py-4 flex justify-between items-center sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-xl shadow-lg">
            <ShieldCheck className="text-white w-5 h-5" />
          </div>
          <h1 className="text-lg font-bold tracking-tight text-slate-800">DeltaGuard <span className="text-indigo-600">Pro</span></h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-full text-xs font-bold text-slate-500">
            <Activity size={14} className="text-green-500" />
            Market Active
          </div>
        </div>
      </nav>

      <main className="p-6 max-w-6xl mx-auto space-y-6">
        <div className="grid grid-cols-12 gap-6">
          
          {/* Sidebar - Cleaned Controls */}
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-6">
                <Settings size={18} className="text-slate-400" />
                <h2 className="font-bold text-slate-700">Position Configuration</h2>
              </div>
              
              <div className="space-y-6">
                {/* Ticker Selection */}
                <div>
                  <label className="text-xs font-bold text-slate-400 uppercase block mb-3">Target Ticker</label>
                  <div className="grid grid-cols-2 gap-2">
                    {['SPY', 'QQQ', 'DIA', 'IWM'].map(t => (
                      <button 
                        key={t}
                        onClick={() => setTicker(t)}
                        className={`py-2.5 rounded-xl font-bold text-sm transition-all ${ticker === t ? 'bg-indigo-600 text-white shadow-md' : 'bg-slate-50 text-slate-600 hover:bg-slate-100 border border-transparent hover:border-slate-200'}`}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Fixed Constraints Information */}
                <div className="space-y-3 pt-2">
                  <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <div className="flex items-center gap-3">
                      <Layers size={16} className="text-indigo-500" />
                      <span className="text-sm font-semibold text-slate-600">Position</span>
                    </div>
                    <span className="text-sm font-bold text-slate-900">ATM Call</span>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <div className="flex items-center gap-3">
                      <Calendar size={16} className="text-indigo-500" />
                      <span className="text-sm font-semibold text-slate-600">Expiry</span>
                    </div>
                    <span className="text-sm font-bold text-slate-900">30 Days</span>
                  </div>
                </div>

                <div className="space-y-3">
                  <label className="text-xs font-bold text-slate-400 uppercase block">
                    Portfolio delta (share-equivalent)
                  </label>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={portfolioDeltaInput}
                    onChange={(e) => setPortfolioDeltaInput(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-800"
                  />
                  <p className="text-[10px] text-slate-400 leading-snug">
                    Use total delta in shares (e.g. option delta 0.48 × 100 contracts → 48).
                  </p>
                </div>

                <div className="space-y-3">
                  <label className="text-xs font-bold text-slate-400 uppercase block">
                    Current hedge (shares, negative if short)
                  </label>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={currentHedgeInput}
                    onChange={(e) => setCurrentHedgeInput(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-800"
                  />
                </div>

                <button 
                  onClick={handleUpdate}
                  disabled={isUpdating}
                  className="w-full bg-slate-900 hover:bg-black text-white font-bold py-4 rounded-xl flex items-center justify-center gap-3 transition-all disabled:opacity-70 shadow-lg shadow-slate-200"
                >
                  {isUpdating ? <RefreshCcw className="animate-spin" size={18} /> : <LayoutDashboard size={18} />}
                  {isUpdating ? 'Analyzing Market...' : 'Refresh Recommendation'}
                </button>
                {predictionError && (
                  <p className="text-xs text-rose-600 font-semibold">{predictionError}</p>
                )}
              </div>
            </div>

            {/* Simple Price Card */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
              <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1 text-[10px]">Underlying Price</p>
              <div className="flex items-baseline gap-2">
                <h3 className="text-3xl font-black text-slate-900 tracking-tighter">${displayedPrice.toLocaleString()}</h3>
                <span className={`text-xs font-bold ${changeColor}`}>{changeText}</span>
              </div>
              {isCsvBackedTicker && isQuoteLoading && (
                <p className="text-xs text-slate-400 font-semibold mt-2">Loading {ticker} quote...</p>
              )}
              {isCsvBackedTicker && quoteError && (
                <p className="text-xs text-rose-600 font-semibold mt-2">{quoteError}</p>
              )}
              {isCsvBackedTicker && historyError && (
                <p className="text-xs text-rose-600 font-semibold mt-2">{historyError}</p>
              )}
            </div>
          </div>

          {/* Main Content */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            
            {/* Recommendation Area */}
            <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-8">
              <div className="flex flex-col md:flex-row items-center gap-10">
                <div className="flex-1 text-center md:text-left">
                  <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-[10px] font-black uppercase tracking-widest mb-4">
                    <BrainCircuit size={12} /> ML Optimization
                  </div>
                  <h2 className="text-7xl font-black text-slate-900 tracking-tight">
                    {recomHedgePct != null ? `${recomHedgePct}%` : '—'}{' '}
                    <span className="text-slate-300">Hedge</span>
                  </h2>
                  <p className="text-slate-500 mt-4 text-sm font-medium leading-relaxed max-w-sm">
                    SPY-trained XGBoost hedge ratio for the 30-day ATM template on {ticker}. Refresh to
                    re-run inference.
                  </p>
                  {predictionConfidence !== null && (
                    <p className="text-xs font-semibold text-slate-400 mt-3">
                      Model confidence: {(predictionConfidence * 100).toFixed(1)}%
                    </p>
                  )}
                </div>

                <div className="w-full md:w-72 rounded-2xl border border-slate-100 bg-slate-50 p-4 text-left text-xs space-y-2 text-slate-600">
                  <p className="font-bold text-slate-500 uppercase tracking-wide">After refresh</p>
                  {recommendation ? (
                    <>
                      <div className="flex justify-between gap-2">
                        <span>Predicted hedge ratio</span>
                        <span className="font-mono font-bold text-slate-900">
                          {Number(recommendation.predicted_hedge_ratio_bucket).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between gap-2">
                        <span>Portfolio delta</span>
                        <span className="font-mono font-bold text-slate-900">
                          {Number(recommendation.portfolio_delta).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between gap-2">
                        <span>Target hedge shares</span>
                        <span className="font-mono font-bold text-slate-900">
                          {Number(recommendation.target_hedge_shares).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between gap-2">
                        <span>Current hedge</span>
                        <span className="font-mono font-bold text-slate-900">
                          {Number(recommendation.current_hedge_shares).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between gap-2 border-t border-slate-200 pt-2 mt-2">
                        <span>Adjustment</span>
                        <span className="font-mono font-bold text-indigo-700">
                          {Number(recommendation.shares_to_trade).toFixed(2)} sh
                        </span>
                      </div>
                      <p className="text-slate-500 pt-1 capitalize">{recommendation.action}</p>
                    </>
                  ) : (
                    <p className="text-slate-400">Click refresh to load model output.</p>
                  )}
                </div>
              </div>

              {/* ML Model Recommendation Label */}
              <div className="mt-10 p-7 bg-slate-900 rounded-3xl flex flex-col md:flex-row items-center justify-between gap-6 text-white shadow-xl">
                <div className="flex items-center gap-5">
                  <div className="w-14 h-14 rounded-2xl bg-indigo-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                    <Zap className="text-white" size={28} />
                  </div>
                  <div>
                    <p className="text-xs font-bold opacity-50 uppercase tracking-widest mb-1">ML Model Recommendation</p>
                    <p className="text-2xl font-bold capitalize">
                      {recommendation?.action
                        ? `${recommendation.action} (${ticker})`
                        : 'Refresh to get trade instruction'}
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleExecute}
                  disabled={isExecuting || !recommendation || Math.round(Math.abs(recommendation.shares_to_trade)) === 0}
                  className="bg-white text-slate-900 px-10 py-4 rounded-2xl font-black text-sm hover:bg-indigo-50 transition-all active:scale-95 shadow-xl disabled:opacity-70"
                >
                  {isExecuting ? 'EXECUTING...' : 'EXECUTE'}
                </button>
              </div>
              {executeMessage && (
                <p className="mt-4 text-xs font-semibold text-emerald-600">{executeMessage}</p>
              )}
              {executeError && (
                <p className="mt-4 text-xs font-semibold text-rose-600">{executeError}</p>
              )}
            </div>

            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
              <h3 className="font-bold text-slate-700 text-sm mb-4">Recent Paper Trades</h3>
              <div className="space-y-3">
                {paperOrders.length === 0 && (
                  <p className="text-xs font-medium text-slate-400">No paper orders yet.</p>
                )}
                {paperOrders.slice().reverse().map((order) => (
                  <div key={order.order_id} className="flex items-center justify-between text-xs border border-slate-100 rounded-xl p-3 bg-slate-50">
                    <div className="font-semibold text-slate-700">
                      {order.side?.toUpperCase()} {order.quantity} {order.ticker}
                    </div>
                    <div className="text-slate-500">
                      ${Number(order.filled_price).toFixed(2)} · {order.status}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Simple Visuals */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                <h3 className="font-bold text-slate-700 text-sm mb-6 flex items-center gap-2">
                  <TrendingUp size={16} className="text-indigo-500" />
                  Asset Performance
                </h3>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                      <XAxis dataKey="date" hide />
                      <YAxis domain={['auto', 'auto']} hide />
                      <Area type="monotone" dataKey="close" stroke="#6366f1" strokeWidth={3} fill="#6366f1" fillOpacity={0.05} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                <h3 className="font-bold text-slate-700 text-sm mb-6 flex items-center gap-2">
                  <Activity size={16} className="text-slate-400" />
                  Hedge Intensity History
                </h3>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                      <XAxis dataKey="date" hide />
                      <YAxis hide />
                      <Bar dataKey="hedge_intensity" radius={[6, 6, 6, 6]}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.hedge_intensity >= 75 ? '#6366f1' : '#f1f5f9'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default HedgeDashboard;