import { PhoneCall, PhoneForwarded, PhoneIncoming, Activity } from 'lucide-react';

const TopCards = ({ data }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      
      {/* Active Calls Card */}
      <div className="bg-emerald-500 text-white rounded-xl shadow-md p-6 border-b-4 border-emerald-700">
        <div className="flex justify-between items-start">
          <div>
            <p className="text-emerald-100 font-medium">Active Calls</p>
            <h3 className="text-4xl font-bold mt-2">{data?.active_calls || 0}</h3>
          </div>
          <div className="bg-emerald-400/30 p-3 rounded-lg">
            <Activity className="w-6 h-6 animate-pulse" />
          </div>
        </div>
        <p className="text-sm mt-4 text-emerald-100">Currently in progress</p>
      </div>

      {/* Total Calls Card */}
      <div className="bg-fuchsia-500 text-white rounded-xl shadow-md p-6 border-b-4 border-fuchsia-700">
        <div className="flex justify-between items-start">
          <div>
            <p className="text-fuchsia-100 font-medium">Total Calls</p>
            <h3 className="text-4xl font-bold mt-2">{data?.total_calls || 0}</h3>
          </div>
          <div className="bg-fuchsia-400/30 p-3 rounded-lg">
            <PhoneCall className="w-6 h-6" />
          </div>
        </div>
        <p className="text-sm mt-4 text-fuchsia-100">All time records</p>
      </div>

      {/* Connected Card */}
      <div className="bg-indigo-500 text-white rounded-xl shadow-md p-6 border-b-4 border-indigo-700">
        <div className="flex justify-between items-start">
          <div>
            <p className="text-indigo-100 font-medium">Connected</p>
            <h3 className="text-4xl font-bold mt-2">{data?.connected_calls || 0}</h3>
          </div>
          <div className="bg-indigo-400/30 p-3 rounded-lg">
            <PhoneIncoming className="w-6 h-6" />
          </div>
        </div>
        <p className="text-sm mt-4 text-indigo-100">Successful connections</p>
      </div>

      {/* Success Rate Card */}
      <div className="bg-cyan-500 text-white rounded-xl shadow-md p-6 border-b-4 border-cyan-700">
        <div className="flex justify-between items-start">
          <div>
            <p className="text-cyan-100 font-medium">Success Rate</p>
            <h3 className="text-4xl font-bold mt-2">{data?.success_rate || 0}%</h3>
          </div>
          <div className="bg-cyan-400/30 p-3 rounded-lg">
            <PhoneForwarded className="w-6 h-6" />
          </div>
        </div>
        <p className="text-sm mt-4 text-cyan-100">Connection percentage</p>
      </div>

    </div>
  );
};

export default TopCards;