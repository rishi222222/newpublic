import { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { LogOut, User } from 'lucide-react';
import LoginPage from './LoginPage';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Classifier from './pages/Classifier';
import Generator from './pages/Generator';
import Smiles from './pages/Smiles';

interface User {
  id: string;
  username: string;
}

// Feature pages contain their own UI and API calls now

function App() {
  const API = (import.meta as any).env?.VITE_API_BASE || (import.meta as any).env?.VITE_API_URL || 'http://localhost:5001';
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const navigate = useNavigate();

  // Check authentication status on component mount
  useEffect(() => {
    checkAuthStatus();
    // Check for Google OAuth callback
    checkGoogleAuthCallback();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const response = await fetch(`${API}/api/check-auth`, {
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.authenticated && data.user) {
          setUser(data.user);
        }
      }
    } catch (err) {
      console.error('Auth check failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const checkGoogleAuthCallback = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const authStatus = urlParams.get('auth');
    const username = urlParams.get('user');
    
    if (authStatus === 'success' && username) {
      // Google OAuth was successful
      setUser({ id: username, username: username });
      navigate('/home', { replace: true });
      
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    } else if (authStatus === 'error') {
      const error = urlParams.get('error');
      console.error('Google OAuth error:', error);
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  };

  const handleLogin = (userData: User) => {
    setUser(userData);
    navigate('/home', { replace: true });
  };

  const handleLogout = async () => {
    try {
      await fetch(`${API}/api/logout`, {
        method: 'POST',
        credentials: 'include',
      });
      setUser(null);
      navigate('/', { replace: true });
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Feature UI moved into routed pages

  

  // Show loading screen while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 flex items-center justify-center animate-fadeIn">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  // Show login page if not authenticated
  if (!user) {
    return (
      <div className="animate-fadeIn">
        <LoginPage onLogin={handleLogin} />
      </div>
    );
  }

  // Main application
  return (
    <div className={`min-h-screen relative overflow-hidden transition-all duration-500 ease-in-out ${isDarkMode ? 'bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-950' : 'bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900'}`}>
      {/* Background particles effect */}
      <div className="absolute inset-0 overflow-hidden">
        <div className={`absolute top-20 left-20 w-2 h-2 rounded-full opacity-20 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-blue-400' : 'bg-white'}`}></div>
        <div className={`absolute top-40 right-32 w-1 h-1 rounded-full opacity-30 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-purple-400' : 'bg-blue-300'}`}></div>
        <div className={`absolute bottom-32 left-16 w-2 h-2 rounded-full opacity-25 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-teal-400' : 'bg-purple-300'}`}></div>
        <div className={`absolute top-1/3 left-1/4 w-1 h-1 rounded-full opacity-40 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-indigo-400' : 'bg-white'}`}></div>
        <div className={`absolute bottom-20 right-20 w-2 h-2 rounded-full opacity-20 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-cyan-400' : 'bg-blue-200'}`}></div>
        <div className={`absolute top-2/3 right-1/4 w-1 h-1 rounded-full opacity-30 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-violet-400' : 'bg-purple-200'}`}></div>
      </div>

      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Top bar with theme toggle and logout */}
        <div className="absolute top-4 right-4 flex items-center gap-4">
          <div className={`flex items-center gap-2 backdrop-blur-sm rounded-full px-4 py-2 border transition-all duration-300 ${isDarkMode ? 'bg-slate-800/60 border-blue-600 shadow-lg' : 'bg-white/10 border-white/20'}`}>
            <User className={`w-4 h-4 transition-colors duration-300 ${isDarkMode ? 'text-blue-300' : 'text-blue-300'}`} />
            <span className={`text-sm font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-white'}`}>{user.username}</span>
          </div>
          <button onClick={toggleTheme} className={`rounded-full p-2 transition-all duration-300 hover:scale-110 ${isDarkMode ? 'bg-indigo-700 hover:bg-indigo-600 text-indigo-200 shadow-lg' : 'bg-white/10 hover:bg-white/20 text-blue-300'}`}>{isDarkMode ? '☀️' : '🌙'}</button>
          <button onClick={handleLogout} className="bg-red-500 hover:bg-red-600 text-white rounded-full p-2 transition-all duration-300 hover:scale-110 shadow-lg" title="Logout"><LogOut className="w-4 h-4" /></button>
        </div>

        <Navbar username={user.username} onLogout={handleLogout} />

        <div className="flex-1 p-4">
          <Routes>
            <Route path="/home" element={<Home />} />
            <Route path="/classifier" element={<Classifier isDarkMode={isDarkMode} />} />
            <Route path="/generator" element={<Generator isDarkMode={isDarkMode} />} />
            <Route path="/smiles" element={<Smiles isDarkMode={isDarkMode} />} />
            <Route path="*" element={<Navigate to="/home" replace />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}

export default App;