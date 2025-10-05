import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, Square, Loader2, Database, AlertCircle } from 'lucide-react';

// For local development, use the direct backend URL.
// For production, you might use a relative path or an environment variable.
const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  // This state is no longer strictly necessary but can be kept for potential future use.
  const [audioChunks, setAudioChunks] = useState([]); 
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // --- UPDATED FUNCTION ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      // Let the browser use its default recorder settings. We will construct the WAV blob on stop.
      const recorder = new MediaRecorder(stream);
      
      const chunks = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      // This is the most important change. We create the Blob as a WAV file.
      recorder.onstop = async () => {
        // Create the blob with the 'audio/wav' type
        const audioBlob = new Blob(chunks, { type: 'audio/wav' });
        // And tell the backend it's a .wav file by passing the filename
        await handleAudioSubmit(audioBlob, 'recording.wav');
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setAudioChunks(chunks);
      setIsRecording(true);
    } catch (error)
    {
      console.error('Error starting recording:', error);
      addMessage('assistant', 'Error accessing microphone. Please check permissions.', true);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const addMessage = (role, content, isError = false, sqlQuery = null, data = null, type = 'text') => {
    const message = {
      role,
      content,
      isError,
      sqlQuery,
      data,
      type,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, message]);
  };

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    addMessage('user', inputText, false, null, null, 'text');
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/query-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: inputText }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      addMessage('assistant', data.message, false, data.sql_query, data.data, 'text');
    } catch (error) {
      console.error('Error:', error);
      addMessage('assistant', 'Sorry, there was an error processing your request. Please check if the backend server is running.', true);
    } finally {
      setIsLoading(false);
    }
  };

  // --- UPDATED FUNCTION ---
  // Now accepts a filename to use in the FormData
  const handleAudioSubmit = async (audioBlob, filename) => {
    setIsLoading(true);

    try {
      const formData = new FormData();
      // Use the filename passed from the onstop handler (e.g., 'recording.wav')
      formData.append('audio', audioBlob, filename);

      const response = await fetch(`${API_BASE}/query-audio`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      addMessage('assistant', data.message, false, data.sql_query, data.data, 'audio');
    } catch (error) {
      console.error('Error:', error);
      addMessage('assistant', 'Sorry, there was an error processing your audio. Please try again.', true, null, null, 'audio');
    } finally {
      setIsLoading(false);
    }
  };
  
  const formatDataAsTable = (data) => {
    if (!data || data.length === 0) return null;

    const columns = Object.keys(data[0]);

    return (
      <div className="overflow-x-auto mt-2 border border-gray-200 rounded-lg">
        <table className="min-w-full bg-white">
          <thead>
            <tr className="bg-gray-50">
              {columns.map(column => (
                <th key={column} className="px-4 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-b">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {data.map((row, index) => (
              <tr key={index} className="hover:bg-gray-50">
                {columns.map(column => (
                  <td key={column} className="px-4 py-3 text-sm text-gray-900 whitespace-nowrap">
                    {String(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="bg-gray-50 px-4 py-2 text-xs text-gray-500 border-t">
          Showing {data.length} row(s)
        </div>
      </div>
    );
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Database className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">SQL Chatbot</h1>
                <p className="text-sm text-gray-600">Ask questions about your database using text or voice</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500 hidden md:block">
                Tables: users, products, orders, order_items, product_sales, categories
              </div>
              {messages.length > 0 && (
                <button
                  onClick={clearChat}
                  className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  Clear Chat
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-6xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-gray-500 mt-20">
              <Database className="h-16 w-16 mx-auto mb-4 text-gray-300" />
              <p className="text-lg font-medium">Start a conversation with your database</p>
              <p className="text-sm mt-2 max-w-md mx-auto">
                Try asking: "Show me the top 5 products by revenue" or "How many users signed up this week?"
              </p>
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <h3 className="font-medium text-gray-900 mb-2">Example Queries</h3>
                  <ul className="text-sm text-gray-600 space-y-1 text-left">
                    <li>• "Show all users"</li>
                    <li>• "Top selling products"</li>
                    <li>• "Recent orders"</li>
                    <li>• "Total revenue by product"</li>
                  </ul>
                </div>
                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <h3 className="font-medium text-gray-900 mb-2">Voice Commands</h3>
                  <p className="text-sm text-gray-600 text-left">
                    Click the microphone button and speak 
                  </p>
                </div>
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-2xl lg:max-w-3xl rounded-lg px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-none'
                    : message.isError
                    ? 'bg-red-50 text-red-800 border border-red-200'
                    : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none shadow-sm'
                }`}
              >
                <div className="flex items-center space-x-2 mb-2">
                  {message.type === 'audio' && (
                    <span className="text-xs bg-black bg-opacity-20 px-2 py-1 rounded">
                      🎤 Voice
                    </span>
                  )}
                  {message.isError && (
                    <AlertCircle className="h-4 w-4" />
                  )}
                </div>
                
                <p className="whitespace-pre-wrap">{message.content}</p>
                
                {message.sqlQuery && !message.isError && (
                  <div className="mt-3 pt-3 border-t border-gray-200 border-opacity-30">
                    <details className="text-sm">
                      <summary className="cursor-pointer font-medium text-gray-600 hover:text-gray-800">
                         View Generated SQL
                      </summary>
                      <pre className="mt-2 p-3 bg-gray-800 text-gray-100 rounded text-xs overflow-x-auto">
                        {message.sqlQuery}
                      </pre>
                    </details>
                  </div>
                )}
                
                {message.data && !message.isError && formatDataAsTable(message.data)}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg rounded-bl-none px-4 py-3 max-w-3xl shadow-sm">
                <div className="flex items-center space-x-2">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                  <span className="text-gray-600">Processing your query...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 px-4 py-4">
        <div className="max-w-6xl mx-auto">
          <form onSubmit={handleTextSubmit} className="flex space-x-3">
            <div className="flex-1">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Ask a question about your data (e.g., 'show me top products by revenue')..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                disabled={isLoading || isRecording}
              />
            </div>
            
            <button
              type="submit"
              disabled={!inputText.trim() || isLoading || isRecording}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center space-x-2 transition-colors"
            >
              <Send className="h-4 w-4" />
              <span className="hidden sm:inline">Send</span>
            </button>

            <button
              type="button"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isLoading}
              className={`px-6 py-3 rounded-lg flex items-center space-x-2 transition-colors ${
                isRecording
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-gray-600 text-white hover:bg-gray-700'
              } disabled:bg-gray-400 disabled:cursor-not-allowed`}
            >
              {isRecording ? (
                <>
                  <Square className="h-4 w-4" />
                  <span className="hidden sm:inline">Stop</span>
                </>
              ) : (
                <>
                  <Mic className="h-4 w-4" />
                  <span className="hidden sm:inline">Voice</span>
                </>
              )}
            </button>
          </form>
          
          {isRecording && (
            <div className="text-center mt-3">
              <div className="inline-flex items-center space-x-2 bg-red-50 text-red-700 px-3 py-1 rounded-full">
                <div className="h-2 w-2 bg-red-600 rounded-full animate-ping"></div>
                <span className="text-sm font-medium">Recording...</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;