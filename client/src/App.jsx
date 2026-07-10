import { useState, useEffect } from "react";
import QueryInput from "./components/QueryInput";
import ResultsDisplay from "./components/ResultsDisplay";
import SettingsModal from "./components/SettingsModal";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [config, setConfig] = useState(null);
  const [indexingStatus, setIndexingStatus] = useState("");

  useEffect(() => {
    const savedConfig = localStorage.getItem("db_bot_config");
    if (savedConfig) {
      try {
        setConfig(JSON.parse(savedConfig));
      } catch (e) {
        console.error(e);
      }
    }
  }, []);

  const handleConfigSave = (newConfig) => {
    setConfig(newConfig);
  };

  const handleIndexSchema = async () => {
    if (!config || !config.db_host) {
      alert("Please configure your DB connection in settings first.");
      return;
    }
    setIndexingStatus("Indexing schema...");
    try {
      const res = await fetch("http://localhost:8000/index-schema", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Indexing failed");
      setIndexingStatus(`Success: ${data.message}`);
      setTimeout(() => setIndexingStatus(""), 5000);
    } catch (err) {
      setIndexingStatus(`Error: ${err.message}`);
    }
  };

  const handleQuery = async (queryText) => {
    if (!config || !config.db_host) {
      setError("Please configure your Database in the settings first.");
      return;
    }

    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText, config }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Query failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center py-10 px-4 relative">
      <SettingsModal onSave={handleConfigSave} />

      <h1 className="text-3xl font-bold text-gray-800 mb-2">AI Database Analytics</h1>
      <p className="text-gray-500 mb-8 text-center max-w-xl">
        Query your database using natural language. Powered by LangGraph and Groq.
      </p>

      {config && (
        <div className="mb-6 flex items-center space-x-4">
          <button
            onClick={handleIndexSchema}
            className="text-sm bg-indigo-100 text-indigo-700 px-4 py-2 rounded-lg font-medium hover:bg-indigo-200 transition-colors"
          >
            Index Database Schema
          </button>
          {indexingStatus && <span className="text-sm text-gray-600">{indexingStatus}</span>}
        </div>
      )}

      <QueryInput onSubmit={handleQuery} loading={loading} />

      {error && (
        <div className="mt-6 bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-xl w-full max-w-4xl shadow-sm">
          <p className="font-semibold flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            Error
          </p>
          <p className="mt-1 text-sm">{error}</p>
        </div>
      )}

      {result && <ResultsDisplay result={result} />}
    </div>
  );
}
