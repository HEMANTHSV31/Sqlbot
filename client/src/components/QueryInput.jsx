import { useState } from "react";
import { Search } from "lucide-react";

export default function QueryInput({ onSubmit, loading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSubmit(query);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center w-full max-w-2xl bg-white shadow-md rounded-2xl p-4"
    >
      <input
        type="text"
        placeholder="Ask: Show average marks per department..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="flex-1 outline-none px-3 py-2 text-gray-800"
      />
      <button
        type="submit"
        disabled={loading}
        className="bg-blue-600 text-white px-4 py-2 rounded-xl hover:bg-blue-700 disabled:opacity-60"
      >
        {loading ? "Processing..." : <Search size={20} />}
      </button>
    </form>
  );
}
