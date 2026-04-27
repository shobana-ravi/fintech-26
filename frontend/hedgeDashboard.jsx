import React, { useState, useMemo } from 'react';
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
  ArrowRightLeft,
  LayoutDashboard,
  BrainCircuit,
  Calendar,
  Layers,
  Zap
} from 'lucide-react';
import { getHedgeRecommendation } from './services/hedgeApi';

const HedgeDashboard = () => {
  // State for parameters
  const [ticker, setTicker] = useState('SPY');
  const [isUpdating, setIsUpdating] = useState(false);
  const [predictionError, setPredictionError] = useState('');
  const [predictionConfidence, setPredictionConfidence] = useState(null);
  
  // Ticker price data
  const tickerPrices = { 
    'SPY': 512.45, 
    'QQQ': 445.18,
    'DIA': 389.21, 
    'IWM': 204.12 
  };
  const currentPrice = tickerPrices[ticker];
  
  // The Model Output (0, 25, 50, 75, 100)
  const [recomHedge, setRecomHedge] = useState(50); 

  // Fixed Parameters
  const dte = 30; 
  const positionType = 'ATM Call';

  // Execution logic: Convert % into shares to buy/sell
  const sharesToHedge = useMemo(() => {
    // Assumption: ATM Call typically has ~50 delta
    const baseDelta = 50; 
    const targetHedgeAmount = Math.round(Math.abs(baseDelta * (recomHedge / 100)));
    return { count: targetHedgeAmount, action: 'Sell' };
  }, [recomHedge]);

  // Mock historical data for charts
  const chartData = useMemo(() => {
    return Array.from({ length: 15 }, (_, i) => ({
      date: `Day ${i + 1}`,
      price: currentPrice - 5 + Math.random() * 10,
      suggestedHedge: [0, 25, 50, 75, 100][Math.floor(Math.random() * 5)]
    }));
  }, [ticker, currentPrice]);

  const hedgeOptions = [
    { value: 0, label: '0% hedge' },
    { value: 25, label: '25% hedge' },
    { value: 50, label: '50% hedge' },
    { value: 75, label: '75% hedge' },
    { value: 100, label: '100% hedge' },
  ];

  const buildFeaturePayload = () => {
    const strike = Math.round(currentPrice);

    return {
      ticker,
      close: currentPrice,
      return_1d: 0.0,
      return_5d: 0.0,
      realized_vol_20d: 0.2,
      strike,
      T: dte / 365,
      option_price: Math.max(currentPrice * 0.02, 0.01),
      delta: 0.5,
      gamma: 0.15,
      theta: -6.0,
      vega: 9.0
    };
  };

  const handleUpdate = async () => {
    setIsUpdating(true);
    setPredictionError('');

    try {
      const prediction = await getHedgeRecommendation(buildFeaturePayload());
      setRecomHedge(Math.round(Number(prediction.predicted_hedge_ratio_bucket) * 100));
      setPredictionConfidence(prediction.prediction_confidence);
    } catch (error) {
      setPredictionError(error.message || 'Could not fetch recommendation');
    } finally {
      setIsUpdating(false);
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
                <h3 className="text-3xl font-black text-slate-900 tracking-tighter">${currentPrice.toLocaleString()}</h3>
                <span className="text-xs font-bold text-green-600">+0.15%</span>
              </div>
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
                    {recomHedge}% <span className="text-slate-300">Hedge</span>
                  </h2>
                  <p className="text-slate-500 mt-4 text-sm font-medium leading-relaxed max-w-sm">
                    Strategic offset recommended for the 30-day ATM Call on {ticker}. 
                  </p>
                  {predictionConfidence !== null && (
                    <p className="text-xs font-semibold text-slate-400 mt-3">
                      Model confidence: {(predictionConfidence * 100).toFixed(1)}%
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-1 gap-2 w-full md:w-52">
                  {hedgeOptions.map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => setRecomHedge(opt.value)}
                      className={`px-6 py-3 rounded-2xl border-2 transition-all font-bold text-sm ${
                        recomHedge === opt.value 
                        ? 'border-indigo-600 bg-indigo-50 text-indigo-900 shadow-sm ring-1 ring-indigo-600' 
                        : 'border-slate-50 bg-slate-50 text-slate-400 hover:border-slate-200 hover:bg-white'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
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
                    <p className="text-2xl font-bold">{sharesToHedge.action} {sharesToHedge.count} Shares of {ticker}</p>
                  </div>
                </div>
                <button className="bg-white text-slate-900 px-10 py-4 rounded-2xl font-black text-sm hover:bg-indigo-50 transition-all active:scale-95 shadow-xl">
                  EXECUTE
                </button>
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
                      <Area type="monotone" dataKey="price" stroke="#6366f1" strokeWidth={3} fill="#6366f1" fillOpacity={0.05} />
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
                      <Bar dataKey="suggestedHedge" radius={[6, 6, 6, 6]}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.suggestedHedge >= 75 ? '#6366f1' : '#f1f5f9'} />
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