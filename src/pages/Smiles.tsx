import React, { useState } from 'react';
import { Activity, AlertCircle, CheckCircle, Loader } from 'lucide-react';

interface Props { isDarkMode: boolean; }
interface SmilesApiResponse { success: boolean; input_sequence: string; smiles?: string; error?: string; }

const Smiles: React.FC<Props> = ({ isDarkMode }) => {
  const API = (import.meta as any).env?.VITE_API_BASE || (import.meta as any).env?.VITE_API_URL || 'http://localhost:5001';
  const [sequence, setSequence] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [smiles, setSmiles] = useState('');

  const validateSequence = (s: string) => /^[ACDEFGHIKLMNPQRSTVWY]+$/i.test(s.trim());

  const handleGenerate = async () => {
    const s = sequence.trim().toUpperCase();
    if (!s) return setError('Please enter a protein sequence');
    if (s.length < 10) return setError('Sequence too short (min 10)');
    if (!validateSequence(s)) return setError('Invalid amino acid sequence');

    setLoading(true); setError(null); setSmiles('');
    try {
      const res = await fetch(`${API}/generate-smiles`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify({ sequence: s })
      });
      const data: SmilesApiResponse = await res.json();
      if (data.success && data.smiles) setSmiles(data.smiles); else setError(data.error || 'SMILES generation failed');
    } catch {
      setError('Unable to connect to backend');
    } finally { setLoading(false); }
  };

  return (
    <div className="max-w-5xl mx-auto p-4">
      <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-green-950/90 border border-green-600' : 'bg-green-50/90 border border-green-200'}`}>
        <label className={`block text-xl font-semibold mb-4 ${isDarkMode ? 'text-green-100' : 'text-green-800'}`}>Protein Sequence (for SMILES)</label>
        <textarea value={sequence} onChange={e=>setSequence(e.target.value)} placeholder="Enter protein sequence to generate SMILES..." className={`w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none text-base font-mono ${isDarkMode ? 'bg-slate-700 border-green-600 text-green-100 placeholder-green-300' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`} disabled={loading} />
        <div className="mt-4 flex items-center gap-4">
          <button onClick={handleGenerate} disabled={loading || !sequence.trim()} className="bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-3 px-6 rounded-xl flex items-center gap-2">
            {loading ? (<><Loader className="w-5 h-5 animate-spin"/>Generating...</>) : (<><Activity className="w-5 h-5"/>Generate SMILES</>)}
          </button>
          {error && <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-xl"><AlertCircle className="w-4 h-4 text-red-500"/><p className="text-red-700 text-sm">{error}</p></div>}
        </div>

        {smiles && (
          <div className="mt-6 space-y-4">
            <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
              <CheckCircle className="w-5 h-5 text-green-500"/>
              <div>
                <p className="text-green-800 font-semibold">SMILES Generation Complete</p>
                <p className="text-green-600 text-sm">Generated SMILES structure from protein sequence</p>
              </div>
            </div>
            <div className={`p-6 rounded-xl border-2 ${isDarkMode ? 'bg-slate-800/50 border-green-600' : 'bg-white border-green-200'}`}>
              <h4 className={`text-lg font-semibold mb-3 ${isDarkMode ? 'text-green-100' : 'text-green-800'}`}>Generated SMILES Structure</h4>
              <div className={`${isDarkMode ? 'bg-slate-900 text-green-300 border border-slate-600' : 'bg-gray-900 text-green-300 border border-gray-700'} font-mono text-sm p-4 rounded-lg`}>{smiles}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Smiles;
