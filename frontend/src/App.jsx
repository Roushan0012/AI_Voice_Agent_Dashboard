import React, { useState, useEffect } from 'react';
import axios from 'axios';
import TopCards from './components/TopCards';
import Charts from './components/Charts';
import CallTable from './components/CallTable';
import CallButton from './components/CallButton'; // <-- Your new button import

function App() {
  const [summaryData, setSummaryData] = useState({
    active_calls: 0, total_calls: 0, connected_calls: 0, success_rate: 0
  });
  
  const fetchDashboardData = async () => {
    try {
      const summaryRes = await axios.get('http://127.0.0.1:5000/api/call_summary');
      setSummaryData(summaryRes.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    // Refreshes the dashboard data every 5 seconds
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 p-8 font-sans text-gray-800">
      
      {/* --- HEADER SECTION --- */}
      <header className="mb-8 flex justify-between items-center bg-gray-900 text-white p-4 rounded-xl shadow-lg">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          🎙️ AI Voice Agent Dashboard - RupeeQ
        </h1>
        
        {/* Right side of the header containing the button and status badge */}
        <div className="flex items-center gap-4">
          
          {/* Your new glowing Start Call button */}
          <CallButton />

          {/* System Online Badge */}
          <div className="flex items-center gap-2 bg-green-500/20 text-green-400 px-4 py-2 rounded-full border border-green-500/30">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </span>
            <span className="text-sm font-medium">System Online</span>
          </div>
        </div>
      </header>

      {/* --- MAIN CONTENT SECTION --- */}
      <main className="space-y-6 max-w-7xl mx-auto">
        <TopCards data={summaryData} />
        <Charts />
        <CallTable />
      </main>

    </div>
  );
}

export default App;