import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar: React.FC<{ username: string; onLogout: () => void }>=({ username, onLogout })=>{
  const { pathname } = useLocation();
  const item = (to: string, label: string)=>{
    const active = pathname === to;
    return (
      <Link
        to={to}
        className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${active ? 'bg-white/20 border-white/40 text-white' : 'bg-white/10 border-white/20 text-blue-100 hover:bg-white/20'}`}
      >
        {label}
      </Link>
    );
  };
  return (
    <div className="w-full sticky top-0 z-50 backdrop-blur bg-gradient-to-r from-blue-900/60 to-indigo-900/60 border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-white font-bold">T2D Insulin</span>
          <nav className="flex items-center gap-2">
            {item('/home','Home')}
            {item('/classifier','Classifier')}
            {item('/generator','Sequence Generator')}
            {item('/smiles','SMILES Generator')}
          </nav>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-blue-100 text-sm">{username}</span>
          <button onClick={onLogout} className="text-white bg-red-500 hover:bg-red-600 px-3 py-1.5 rounded-md text-sm">Logout</button>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
