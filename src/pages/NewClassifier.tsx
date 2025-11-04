import { Navigate } from 'react-router-dom';

const NewClassifier: React.FC<{ isDarkMode: boolean }> = () => {
  return <Navigate to="/classifier" replace />;
};

export default NewClassifier;
