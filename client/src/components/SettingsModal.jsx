import { useState, useEffect } from "react";
import { Settings, X } from "lucide-react";

export default function SettingsModal({ onSave }) {
  const [isOpen, setIsOpen] = useState(false);
  const [config, setConfig] = useState({
    db_host: "",
    db_user: "",
    db_password: "",
    db_name: "",
    groq_api_key: "",
    pinecone_api_key: "",
    pinecone_index_name: "dynamic_schema_index"
  });

  useEffect(() => {
    const savedConfig = localStorage.getItem("db_bot_config");
    if (savedConfig) {
      try {
        const parsed = JSON.parse(savedConfig);
        setConfig(parsed);
      } catch (e) {
        console.error("Error parsing config", e);
      }
    }
  }, []);

  const handleChange = (e) => {
    setConfig({ ...config, [e.target.name]: e.target.value });
  };

  const handleSave = () => {
    localStorage.setItem("db_bot_config", JSON.stringify(config));
    setIsOpen(false);
    if (onSave) onSave(config);
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed top-4 right-4 bg-white p-2 rounded-full shadow-md text-gray-600 hover:text-gray-900 transition-colors"
        title="Settings"
      >
        <Settings size={24} />
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl w-full max-w-lg p-6 shadow-xl max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Configuration</h2>
              <button onClick={() => setIsOpen(false)} className="text-gray-500 hover:text-gray-800">
                <X size={24} />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2 border-b pb-1">Database Settings</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Host</label>
                    <input type="text" name="db_host" value={config.db_host} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none" />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Database Name</label>
                    <input type="text" name="db_name" value={config.db_name} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none" />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">User</label>
                    <input type="text" name="db_user" value={config.db_user} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none" />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Password</label>
                    <input type="password" name="db_password" value={config.db_password} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none" />
                  </div>
                </div>
              </div>

              <button onClick={handleSave} className="w-full bg-blue-600 text-white font-medium py-3 rounded-xl hover:bg-blue-700 transition-colors mt-6">
                Save Configuration
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
