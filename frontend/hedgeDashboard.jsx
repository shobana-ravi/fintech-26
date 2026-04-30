import React, { useState, useMemo, useEffect } from 'react';
import {
  XAxis, YAxis, CartesianGrid, ResponsiveContainer, AreaChart, Area,
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
import {
  getHedgeRecommendation,
  getQuote,
  getHistory,
  getLatestFeatures,
  placePaperOrder,
  getPaperOrders
} from './services/hedgeApi';

const HedgeDashboard = () => {
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
  const [contractCountInput, setContractCountInput] = useState('1');
  const [portfolioDeltaInput, setPortfolioDeltaInput] = useState('');
  const [currentHedgeInput, setCurrentHedgeInput] = useState('0');
  const [latestFeatureRow, setLatestFeatureRow] = useState(null);
  const [latestFeatureError, setLatestFeatureError] = useState('');

  const tickerPrices = {
    SPY: 512.45,
    QQQ: 445.18,
    DIA: 389.21,
    IWM: 204.12
  };

  const currentPrice = tickerPrices[ticker];
  const csvBackedTickers = new Set(['DIA', 'IWM', 'QQQ', 'SPY']);
  const isCsvBackedTicker = csvBackedTickers.has(ticker);

  const displayedPrice = isCsvBackedTicker && tickerQuote?.price ? tickerQuote.price : currentPrice;
  const displayedChangePct =
    isCsvBackedTicker && tickerQuote?.change_pct !== undefined ? tickerQuote.change_pct : 0.15;
  const changeText = `${displayedChangePct >= 0 ? '+' : ''}${displayedChangePct.toFixed(2)}%`;
  const changeColor = displayedChangePct >= 0 ? 'text-green-600' : 'text-rose-600';

  // Reflect backend value directly: 0.25 -> 25%, 1.0 -> 100%
  const recomHedgePct = useMemo(() => {
    if (!recommendation) return null;
    const raw = Number(recommendation.predicted_hedge_ratio_bucket);
    if (!Number.isFinite(raw)) return null;
    return Math.round(raw * 100);
  }, [recommendation]);

  const fallbackChartData = useMemo(() => {
    return Array.from({ length: 15 }, (_, i) => ({
      date: `Day ${i + 1}`,
      close: displayedPrice,
      hedge_intensity: 0,
    }));
  }, [displayedPrice]);

  const chartData = historyData.length > 0 ? historyData : fallbackChartData;

  useEffect(() => {
    setContractCountInput('1');
    setPortfolioDeltaInput('');
    setCurrentHedgeInput('0');
    setRecommendation(null);
    setPredictionConfidence(null);
    setPredictionError('');
    setExecuteMessage('');
    setExecuteError('');
    setLatestFeatureRow(null);
    setLatestFeatureError('');
  }, [ticker]);

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
        if (!isCancelled) setTickerQuote(quote);
      } catch (error) {
        if (!isCancelled) setQuoteError(error.message || `Could not fetch ${ticker} quote`);
      } finally {
        if (!isCancelled) setIsQuoteLoading(false);
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
        if (!isCancelled) setPaperOrders(Array.isArray(data.orders) ? data.orders : []);
      } catch (_error) {
        if (!isCancelled) setPaperOrders([]);
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

  useEffect(() => {
    if (!isCsvBackedTicker) return;

    let isCancelled = false;
    const loadLatestFeatures = async () => {
      setLatestFeatureError('');
      try {
        const row = await getLatestFeatures(ticker);
        if (!isCancelled) {
          setLatestFeatureRow(row);

          const contractCount = Number(contractCountInput);
          const optionDelta = Number(row?.delta ?? 0);
          if (Number.isFinite(contractCount) && Number.isFinite(optionDelta)) {
            const livePortfolioDelta = optionDelta * 100 * contractCount;
            setPortfolioDeltaInput(String(livePortfolioDelta));
          }
        }
      } catch (error) {
        if (!isCancelled) {
          setLatestFeatureError(error.message || `Could not load latest ${ticker} feature row`);
          setLatestFeatureRow(null);
        }
      }
    };

    loadLatestFeatures();
    return () => {
      isCancelled = true;
    };
  }, [isCsvBackedTicker, ticker, contractCountInput]);

  const handleUpdate = async () => {
    setIsUpdating(true);
    setPredictionError('');
    setRecommendation(null);
    setPredictionConfidence(null);

    try {
      if (isCsvBackedTicker) {
        try {
          setIsQuoteLoading(true);
          setQuoteError('');
          const quote = await getQuote(ticker);
          setTickerQuote(quote);
        } catch (error) {
          setQuoteError(error.message || `Could not fetch ${ticker} quote`);
        } finally {
          setIsQuoteLoading(false);
        }
      }

      const latestFeatures = await getLatestFeatures(ticker);
      setLatestFeatureRow(latestFeatures);

      const contractCount = Number(contractCountInput);
      if (!Number.isFinite(contractCount) || contractCount <= 0) {
        throw new Error('Contract count must be a positive number.');
      }

      const optionDelta = Number(latestFeatures?.delta ?? 0);
      if (!Number.isFinite(optionDelta)) {
        throw new Error('Latest option delta is invalid.');
      }

      const livePortfolioDelta = optionDelta * 100 * contractCount;
      setPortfolioDeltaInput(String(livePortfolioDelta));

      const currentHedge = Number(currentHedgeInput);
      if (!Number.isFinite(currentHedge)) {
        throw new Error('Current hedge shares must be a number.');
      }

      const prediction = await getHedgeRecommendation({
        ticker,
        contract_count: contractCount,
        current_hedge_shares: currentHedge,
        use_latest_portfolio_row: true,
      });

      console.log('prediction from backend', prediction);

      setRecommendation(prediction);
      setPredictionConfidence(prediction.prediction_confidence);

      try {
        const history = await getHistory(ticker, 60);
        setHistoryData(
          (history.points || []).map((point) => ({
            date: point.date,
            close: point.close,
            hedge_intensity: point.hedge_intensity,
          })),
        );
      } catch (_e) {
        // ignore
      }
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
        `Paper order filled: ${order.side.toUpperCase()} ${order.quantity} ${order.ticker} @ $${Number(order.filled_price).toFixed(2)}`
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
      <nav className="bg-white border-b border-slate-200 px-8 py-4 flex justify-between items-center sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-xl shadow-lg">
            <ShieldCheck className="text-white w-5 h-5" />
          </div>
          <h1 className="text-lg font-bold tracking-tight text-slate-800">
            DeltaGuard <span className="text-indigo-600">Pro</span>
          </h1>
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
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-6">
                <Settings size={18} className="text-slate-400" />
                <h2 className="font-bold text-slate-700">Position Configuration</h2>
              </div>

              <div className="space-y-6">
                <div>
                  <label className="text-xs font-bold text-slate-400 uppercase block mb-3">Target Ticker</label>
                  <div className="grid grid-cols-2 gap-2">
                    {['SPY', 'QQQ', 'DIA', 'IWM'].map(t => (
                      <button
                        key={t}
                        onClick={() => setTicker(t)}
                        className={`py-2.5 rounded-xl font-bold text-sm transition-all ${
                          ticker === t
                            ? 'bg-indigo-600 text-white shadow-md'
                            : 'bg-slate-50 text-slate-600 hover:bg-slate-100 border border-transparent hover:border-slate-200'
                        }`}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>

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
                    Contract count
                  </label>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={contractCountInput}
                    onChange={(e) => setContractCountInput(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-800"
                  />
                  <p className="text-[10px] text-slate-400 leading-snug">
                    Portfolio delta is computed as option delta × 100 × contract count.
                  </p>
                </div>

                <div className="space-y-3">
                  <label className="text-xs font-bold text-slate-400 uppercase block">
                    Live portfolio delta (computed)
                  </label>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={portfolioDeltaInput}
                    readOnly
                    className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-800 bg-slate-50"
                  />
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
                {latestFeatureError && (
                  <p className="text-xs text-rose-600 font-semibold">{latestFeatureError}</p>
                )}
              </div>
            </div>

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

          <div className="col-span-12 lg:col-span-8 space-y-6">
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
                    Uses latest backend row for market features and user contract count for live portfolio delta.
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
                        <span>Raw backend bucket</span>
                        <span className="font-mono font-bold text-slate-900">
                          {String(recommendation.predicted_hedge_ratio_bucket)}
                        </span>
                      </div>
                      {recommendation.soft_hedge_ratio_bucket !== undefined && (
                        <div className="flex justify-between gap-2">
                          <span>Soft bucket</span>
                          <span className="font-mono font-bold text-slate-900">
                            {Number(recommendation.soft_hedge_ratio_bucket).toFixed(4)}
                          </span>
                        </div>
                      )}
                      <div className="flex justify-between gap-2">
                        <span>Model used</span>
                        <span className="font-mono font-bold text-slate-900">
                          {recommendation.model_used || recommendation.ticker}
                        </span>
                      </div>
                      <div className="flex justify-between gap-2">
                        <span>Contract count</span>
                        <span className="font-mono font-bold text-slate-900">
                          {Number(recommendation.contract_count ?? contractCountInput).toFixed(2)}
                        </span>
                      </div>
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

            {latestFeatureRow && (
              <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                <h3 className="font-bold text-slate-700 text-sm mb-4">Latest Backend Feature Row</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                  <div className="rounded-xl bg-slate-50 p-3">
                    <div className="text-slate-400">Spot</div>
                    <div className="font-bold text-slate-900">
                      {Number(latestFeatureRow.spot_today ?? latestFeatureRow.close ?? 0).toFixed(2)}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-3">
                    <div className="text-slate-400">Option Delta</div>
                    <div className="font-bold text-slate-900">
                      {Number(latestFeatureRow.delta ?? 0).toFixed(4)}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-3">
                    <div className="text-slate-400">Realized Vol</div>
                    <div className="font-bold text-slate-900">
                      {Number(latestFeatureRow.realized_vol_20d ?? 0).toFixed(4)}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-3">
                    <div className="text-slate-400">Strike</div>
                    <div className="font-bold text-slate-900">
                      {Number(latestFeatureRow.strike ?? 0).toFixed(0)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default HedgeDashboard;