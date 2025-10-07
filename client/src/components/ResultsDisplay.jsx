import ChartView from "./ChartView";

export default function ResultsDisplay({ result }) {
  if (!result) return null;

  return (
    <div className="w-full max-w-5xl mt-8 bg-white shadow-lg rounded-2xl p-6">
      <h2 className="text-lg font-semibold text-gray-700 mb-3">
        SQL Query:
      </h2>
      <pre className="bg-gray-900 text-green-400 p-3 rounded-md overflow-auto text-sm">
        {result.sql}
      </pre>

      {result.is_chart ? (
        <div className="mt-6">
          <ChartView data={result.rows} columns={result.columns} chartType={result.chart_type} />
        </div>
      ) : (
        <div className="overflow-x-auto mt-6">
          <table className="min-w-full border border-gray-300 text-sm">
            <thead className="bg-gray-200">
              <tr>
                {result.columns.map((col) => (
                  <th key={col} className="px-3 py-2 border-b border-gray-300">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  {result.columns.map((col) => (
                    <td key={col} className="px-3 py-2 border-b border-gray-200 text-center">
                      {row[col]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
