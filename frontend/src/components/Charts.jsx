import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, 
  BarChart, Bar, XAxis, YAxis, CartesianGrid 
} from 'recharts';

const Charts = () => {
  // 1. State to hold our chart data
  const [pieData, setPieData] = useState([]);
  const [barData, setBarData] = useState([]);

  // 2. Fetch the data from your Flask API
  useEffect(() => {
    const fetchCharts = async () => {
      try {
        const pieRes = await axios.get('http://127.0.0.1:5000/api/call_status_pie');
        const barRes = await axios.get('http://127.0.0.1:5000/api/call_outcomes_bar');
        
        setPieData(pieRes.data);
        setBarData(barRes.data);
      } catch (error) {
        console.error("Error fetching chart data:", error);
      }
    };

    fetchCharts();
    // Refresh every 5 seconds to show live updates!
    const interval = setInterval(fetchCharts, 5000);
    return () => clearInterval(interval);
  }, []);

  // Colors for the charts
  const PIE_COLORS = ['#10b981', '#ef4444']; // Emerald (Connected), Red (Failed)
  const BAR_COLORS = ['#3b82f6', '#9ca3af', '#6366f1']; // Blue, Gray, Indigo

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
      
      {/* Chart 1: Call Status Distribution (Donut Chart) */}
      <div className="bg-white p-6 rounded-xl shadow-md">
        <h2 className="text-lg font-bold text-gray-700 mb-4 border-b pb-2">Call Status Distribution</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        {/* Custom Legend */}
        <div className="flex justify-center gap-4 mt-2">
          {pieData.map((entry, index) => (
            <div key={entry.name} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: PIE_COLORS[index] }}></div>
              <span className="text-sm text-gray-600">{entry.name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Chart 2: Call Outcomes (Bar Chart) */}
      <div className="bg-white p-6 rounded-xl shadow-md">
        <h2 className="text-lg font-bold text-gray-700 mb-4 border-b pb-2">Call Outcomes</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
              <XAxis dataKey="name" axisLine={false} tickLine={false} />
              <YAxis allowDecimals={false} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: '#f3f4f6' }} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {barData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
};

export default Charts;