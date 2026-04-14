import { useState, useEffect } from 'react';
import axios from 'axios';
import { Search, Eye, X } from 'lucide-react';

const CallTable = () => {
  const [calls, setCalls] = useState([]);
  const [selectedCall, setSelectedCall] = useState(null); // For the Modal
  
  // Filter States
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('All Status');

  const fetchCalls = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/api/calls', {
        params: {
          search: searchTerm,
          status: statusFilter !== 'All Status' ? statusFilter : ''
        }
      });
      setCalls(response.data);
    } catch (error) {
      console.error("Error fetching calls:", error);
    }
  };

  // FIXED: Fetch immediately on load, AND auto-refresh every 5 seconds!
  useEffect(() => {
    fetchCalls();
    const interval = setInterval(fetchCalls, 5000);
    return () => clearInterval(interval);
  }, [searchTerm, statusFilter]);

  // Helper for Sentiment Color
  const getSentimentColor = (score) => {
    if (score >= 0.65) return 'bg-green-100 text-green-700';
    if (score >= 0.4) return 'bg-yellow-100 text-yellow-700';
    return 'bg-red-100 text-red-700';
  };

  // Helper for clean date formatting
  const formatTime = (timeString) => {
    if (!timeString) return 'N/A';
    return new Date(timeString).toLocaleString('en-IN', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-md mt-6">
      
      {/* Top Filter Bar */}
      <div className="flex flex-col md:flex-row justify-between items-end gap-4 mb-6 bg-slate-50 p-4 rounded-lg border border-slate-100">
        <div className="flex flex-col gap-1 w-full md:w-1/3">
          <label className="text-xs font-semibold text-gray-500 uppercase">Search Transcript</label>
          <div className="relative">
            <input 
              type="text" 
              placeholder="Search in transcripts or names..." 
              className="w-full pl-10 pr-4 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Search className="absolute left-3 top-2.5 text-gray-400 w-5 h-5" />
          </div>
        </div>

        <div className="flex flex-col gap-1 w-full md:w-1/4">
          <label className="text-xs font-semibold text-gray-500 uppercase">Status</label>
          <select 
            className="w-full p-2 border rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option>All Status</option>
            <option>Connected</option>
            <option>Ended</option>
            <option>Failed</option>
          </select>
        </div>

        <button 
          onClick={fetchCalls}
          className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-md font-medium transition-colors flex items-center gap-2"
        >
          <Search className="w-4 h-4" /> LOAD
        </button>
      </div>

      {/* The Table */}
      <h2 className="text-lg font-bold text-gray-700 mb-4 border-b pb-2">Detailed Call Records</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-gray-50 text-gray-600 text-sm border-b">
              <th className="p-3 font-semibold">Customer</th>
              <th className="p-3 font-semibold">Phone</th>
              <th className="p-3 font-semibold">Start Time</th>
              <th className="p-3 font-semibold">Duration</th>
              <th className="p-3 font-semibold">Status</th>
              <th className="p-3 font-semibold">Outcome</th>
              <th className="p-3 font-semibold">Sentiment</th>
              <th className="p-3 font-semibold text-center">Actions</th>
            </tr>
          </thead>
          <tbody>
            {calls.map((call) => (
              <tr key={call.id} className="border-b hover:bg-gray-50 transition-colors">
                <td className="p-3 font-medium text-gray-800">{call.customer}</td>
                <td className="p-3 text-gray-600">{call.phone}</td>
                
                {/* FIXED: Formatted Start Time */}
                <td className="p-3 text-gray-600">{formatTime(call.start_time)}</td>
                
                <td className="p-3 text-gray-600">{call.duration || 'N/A'}</td>
                
                <td className="p-3">
                  <span className={`px-2 py-1 rounded-full text-xs font-bold ${call.status === 'connected' ? 'bg-emerald-100 text-emerald-700' : 'bg-gray-200 text-gray-700'}`}>
                    {call.status ? call.status.toUpperCase() : 'UNKNOWN'}
                  </span>
                </td>
                <td className="p-3">
                  <span className="px-2 py-1 rounded bg-blue-100 text-blue-700 text-xs font-bold">
                    {call.outcome}
                  </span>
                </td>
                <td className="p-3">
                  <span className={`px-2 py-1 rounded text-xs font-bold ${getSentimentColor(call.sentiment)}`}>
                    {Math.round((call.sentiment || 0) * 100)}%
                  </span>
                </td>
                <td className="p-3 text-center">
                  <button 
                    onClick={() => setSelectedCall(call)}
                    className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors"
                    title="View Details"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
            {calls.length === 0 && (
              <tr>
                <td colSpan="8" className="p-6 text-center text-gray-500">No calls found matching your filters.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* The Details Modal */}
      {selectedCall && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col">
            
            {/* Modal Header */}
            <div className="flex justify-between items-center p-4 border-b bg-gray-50 rounded-t-xl">
              <h3 className="text-xl font-bold text-gray-800">Call Details</h3>
              <button onClick={() => setSelectedCall(null)} className="text-gray-500 hover:text-red-500 transition-colors">
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                <div><span className="font-semibold text-gray-500 block text-xs uppercase">Customer</span> <span className="font-medium">{selectedCall.customer}</span></div>
                <div><span className="font-semibold text-gray-500 block text-xs uppercase">Phone</span> <span className="font-medium">{selectedCall.phone}</span></div>
                <div><span className="font-semibold text-gray-500 block text-xs uppercase">Date</span> <span>{formatTime(selectedCall.start_time)}</span></div>
                <div><span className="font-semibold text-gray-500 block text-xs uppercase">Duration</span> <span>{selectedCall.duration || 'N/A'}</span></div>
                <div>
                  <span className="font-semibold text-gray-500 block text-xs uppercase mb-1">Outcome</span> 
                  <span className="px-2 py-1 rounded bg-blue-100 text-blue-700 text-xs font-bold">{selectedCall.outcome}</span>
                </div>
                <div>
                  <span className="font-semibold text-gray-500 block text-xs uppercase mb-1">Sentiment</span> 
                  <span className={`px-2 py-1 rounded text-xs font-bold ${getSentimentColor(selectedCall.sentiment)}`}>
                    {Math.round((selectedCall.sentiment || 0) * 100)}%
                  </span>
                </div>
              </div>

              <div className="border-t pt-4">
                <h4 className="font-bold text-gray-700 mb-2 flex items-center gap-2">
                  Full Transcript
                </h4>
                <div className="bg-slate-50 p-4 rounded-lg border text-sm text-gray-700 whitespace-pre-wrap h-64 overflow-y-auto font-mono leading-relaxed shadow-inner">
                  {selectedCall.transcript || "No transcript available for this call."}
                </div>
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
};

export default CallTable;