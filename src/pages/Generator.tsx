import React, { useState } from 'react';
import { Activity, AlertCircle, CheckCircle, Loader } from 'lucide-react';

interface Props { isDarkMode: boolean; }
interface GeneratedSequence { sequence: string; average_probability: number; levenshtein: number; hamming: number; cosine: number; pearson: number; }
interface SequenceApiResponse { success: boolean; input_sequence: string; generated_sequences: GeneratedSequence[]; error?: string; }

const Generator: React.FC<Props> = ({ isDarkMode }) => {
  const API = (import.meta as any).env?.VITE_API_BASE || (import.meta as any).env?.VITE_API_URL || 'http://localhost:5001';
  const [sequence, setSequence] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [items, setItems] = useState<GeneratedSequence[]>([]);

  const validateSequence = (s: string) => /^[ACDEFGHIKLMNPQRSTVWY]+$/i.test(s.trim());

  const handleGenerate = async () => {
    const s = sequence.trim().toUpperCase();
    if (!s) return setError('Please enter a protein sequence');
    if (s.length < 10) return setError('Sequence too short (min 10)');
    if (!validateSequence(s)) return setError('Invalid amino acid sequence');

    setLoading(true); setError(null); setItems([]);
    try {
      const res = await fetch(`${API}/generate-sequences`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify({ sequence: s })
      });
      const data: SequenceApiResponse = await res.json();
      if (data.success && data.generated_sequences) setItems(data.generated_sequences); else setError(data.error || 'Generation failed');
    } catch {
      setError('Unable to connect to backend');
    } finally { setLoading(false); }
  };

  return (
    <div className="max-w-6xl mx-auto p-4">
      <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-purple-950/80 border border-purple-600' : 'bg-purple-50/90 border border-purple-100'}`}>
        <label className={`block text-xl font-semibold mb-4 ${isDarkMode ? 'text-purple-100' : 'text-purple-800'}`}>Input Protein Sequence for Generation</label>
        <textarea value={sequence} onChange={e=>setSequence(e.target.value)} placeholder="Enter protein sequence to generate new variants..." className={`w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none text-base font-mono ${isDarkMode ? 'bg-slate-700 border-purple-600 text-purple-100 placeholder-purple-300' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`} disabled={loading} />
        <div className="mt-4 flex items-center gap-4">
          <button onClick={handleGenerate} disabled={loading || !sequence.trim()} className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-3 px-6 rounded-xl flex items-center gap-2">
            {loading ? (<><Loader className="w-5 h-5 animate-spin"/>Generating...</>) : (<><Activity className="w-5 h-5"/>Generate Sequences</>)}
          </button>
          {error && <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-xl"><AlertCircle className="w-4 h-4 text-red-500"/><p className="text-red-700 text-sm">{error}</p></div>}
        </div>

        {items.length > 0 && (
          <div className="mt-6 space-y-6">
            <div className="flex items-center gap-3 p-4 bg-purple-50 border border-purple-200 rounded-xl">
              <CheckCircle className="w-5 h-5 text-purple-500"/>
              <div>
                <p className="text-purple-800 font-semibold">Sequence Generation Complete</p>
                <p className="text-purple-600 text-sm">Generated {items.length} sequences with similarity analysis</p>
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {items.map((seq, index) => (
                <div key={index} className={`p-6 rounded-xl border-2 ${isDarkMode ? 'bg-slate-800/50 border-purple-600 hover:border-purple-500' : 'bg-white border-purple-200 hover:border-purple-300'}`}>
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${index===0?'bg-yellow-500':index===1?'bg-gray-400':index===2?'bg-orange-500':'bg-blue-500'}`}>{index+1}</div>
                      <h4 className={`${isDarkMode?'text-purple-100':'text-purple-800'} text-lg font-semibold`}>Sequence #{index+1}</h4>
                    </div>
                    <div className={`${isDarkMode?'bg-slate-900 text-green-300 border border-slate-600':'bg-gray-900 text-green-300 border border-gray-700'} font-mono text-sm p-4 rounded-lg`}>{seq.sequence}</div>
                    <div className={`${isDarkMode?'bg-purple-900/50 text-purple-200':'bg-purple-100 text-purple-800'} text-center p-4 rounded-lg`}>
                      <div className="text-2xl font-bold">{(seq.average_probability*100).toFixed(1)}%</div>
                      <div className={`${isDarkMode?'text-purple-300':'text-purple-600'} text-sm`}>Average Probability</div>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className={`${isDarkMode?'bg-slate-600 text-blue-200':'bg-gray-100 text-gray-700'} p-3 rounded-lg text-center`}>
                        {(seq.levenshtein*100).toFixed(1)}%<div className={`${isDarkMode?'text-blue-300':'text-gray-500'}`}>Levenshtein</div>
                      </div>
                      <div className={`${isDarkMode?'bg-slate-600 text-green-200':'bg-gray-100 text-gray-700'} p-3 rounded-lg text-center`}>
                        {(seq.hamming*100).toFixed(1)}%<div className={`${isDarkMode?'text-green-300':'text-gray-500'}`}>Hamming</div>
                      </div>
                      <div className={`${isDarkMode?'bg-slate-600 text-orange-200':'bg-gray-100 text-gray-700'} p-3 rounded-lg text-center`}>
                        {(seq.cosine*100).toFixed(1)}%<div className={`${isDarkMode?'text-orange-300':'text-gray-500'}`}>Cosine</div>
                      </div>
                      <div className={`${isDarkMode?'bg-slate-600 text-pink-200':'bg-gray-100 text-gray-700'} p-3 rounded-lg text-center`}>
                        {(seq.pearson*100).toFixed(1)}%<div className={`${isDarkMode?'text-pink-300':'text-gray-500'}`}>Pearson</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Generator;
