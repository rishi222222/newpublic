import React, { useState } from 'react';
import { Send, AlertCircle, CheckCircle, Loader } from 'lucide-react';

interface Props { isDarkMode: boolean; }
interface PredictionResult { prediction: string; confidence: number; probabilities: Record<string, number>; }
interface ApiResponse { success: boolean; result?: PredictionResult; processed_sequence?: string; error?: string; }

const Classifier: React.FC<Props> = ({ isDarkMode }) => {
  const API = (import.meta as any).env?.VITE_API_BASE || (import.meta as any).env?.VITE_API_URL || 'http://localhost:5001';
  const [sequence, setSequence] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const validateSequence = (seq: string) => /^[ACDEFGHIKLMNPQRSTVWY]+$/i.test(seq.trim());

  const handlePredict = async () => {
    const s = sequence.trim().toUpperCase();
    if (!s) return setError('Please enter a protein sequence');
    if (s.length < 10) return setError('Sequence too short (min 10)');
    if (!validateSequence(s)) return setError('Invalid amino acid sequence');

    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API}/predict-new-classifier`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify({ sequence: s })
      });
      const data: ApiResponse = await res.json();
      if (data.success && data.result) setResult(data.result); else setError(data.error || 'Prediction failed');
    } catch (e) {
      setError('Unable to connect to backend');
    } finally { setLoading(false); }
  };

  const format = (t: string) => t.replace(/_/g, ' ');

  return (
    <div className="max-w-5xl mx-auto p-4">
      <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800/90 border border-indigo-600' : 'bg-white/95 border border-white/10'}`}>
        <label className={`block text-xl font-semibold mb-4 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>Protein Sequence</label>
        <textarea value={sequence} onChange={e=>setSequence(e.target.value)} placeholder="Enter your protein sequence here..." className={`w-full h-36 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-base font-mono ${isDarkMode ? 'bg-slate-700 border-indigo-600 text-blue-100 placeholder-blue-300' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`} disabled={loading} />
        <div className="mt-4 flex items-center gap-4">
          <button onClick={handlePredict} disabled={loading || !sequence.trim()} className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-3 px-6 rounded-xl flex items-center gap-2">
            {loading ? (<><Loader className="w-5 h-5 animate-spin"/>Analyzing...</>) : (<><Send className="w-5 h-5"/>Predict</>)}
          </button>
          {error && <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-xl"><AlertCircle className="w-4 h-4 text-red-500"/><p className="text-red-700 text-sm">{error}</p></div>}
        </div>
        {result && (
          <div className="mt-6 space-y-4">
            <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
              <CheckCircle className="w-5 h-5 text-green-500"/>
              <div>
                <p className="text-green-800 font-semibold">Prediction Complete</p>
                <p className="text-green-600 text-sm">Confidence {(result.confidence*100).toFixed(1)}%</p>
              </div>
            </div>
            <div className={isDarkMode? 'text-blue-100':'text-gray-800'}>
              <p className="text-2xl font-bold mb-2">{format(result.prediction)}</p>
              <div className="space-y-1">
                {Object.entries(result.probabilities).sort(([,a],[,b])=> (b as number)-(a as number)).map(([k,v])=> (
                  <div key={k} className="flex justify-between text-sm">
                    <span>{format(k)}</span>
                    <span className="font-semibold">{((v as number)*100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Classifier;
