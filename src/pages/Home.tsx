import React from 'react';

const Home: React.FC = () => {
  return (
    <div className="max-w-5xl mx-auto p-6 text-white space-y-8">
      <h1 className="text-4xl font-bold">Welcome to T2D Insulin Predictor</h1>
      <section className="bg-white/10 rounded-xl p-6 border border-white/20">
        <h2 className="text-2xl font-semibold mb-3">What is Insulin?</h2>
        <p className="text-blue-100 leading-relaxed">
          Insulin is a hormone produced by the pancreas that helps glucose from food enter your cells to be used for energy. It plays a critical role in maintaining normal blood glucose levels.
        </p>
      </section>
      <section className="bg-white/10 rounded-xl p-6 border border-white/20">
        <h2 className="text-2xl font-semibold mb-3">Why Insulin Matters in Type 2 Diabetes (T2D)</h2>
        <p className="text-blue-100 leading-relaxed">
          In Type 2 Diabetes, the body becomes resistant to insulin or doesn&apos;t produce enough of it, causing glucose to build up in the blood. Understanding protein sequences and classifying pathogenicity can support research into treatments and personalized interventions.
        </p>
      </section>
      <section className="bg-white/10 rounded-xl p-6 border border-white/20">
        <h2 className="text-2xl font-semibold mb-3">What You Can Do Here</h2>
        <ul className="space-y-2 text-blue-100 list-disc list-inside">
          <li>Use the Classifier to predict pathogenicity from a protein sequence.</li>
          <li>Generate related sequences and compare similarity metrics.</li>
          <li>Generate SMILES strings from protein sequences.</li>
          <li>Sign in with Google or use OTP-based authentication to access features.</li>
        </ul>
      </section>
    </div>
  );
};

export default Home;
