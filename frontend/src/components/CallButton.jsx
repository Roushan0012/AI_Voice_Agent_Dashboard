import { useState } from 'react';
import axios from 'axios';
import { LiveKitRoom, RoomAudioRenderer } from '@livekit/components-react';
import '@livekit/components-styles'; // Required for LiveKit default styles
import { Phone, PhoneOff, Loader2 } from 'lucide-react';

const CallButton = () => {
  const [token, setToken] = useState("");
  const [url, setUrl] = useState("");
  const [isConnecting, setIsConnecting] = useState(false);

  // 1. When the user clicks Start Call
  const startCall = async () => {
    setIsConnecting(true);
    try {
      // Get the VIP token from your Flask backend
      const res = await axios.get('http://127.0.0.1:5000/api/get_token');
      setToken(res.data.token);
      setUrl(res.data.url);
    } catch (error) {
      console.error("Error connecting:", error);
      alert("Make sure your Flask backend is running!");
    } finally {
      setIsConnecting(false);
    }
  };

  // 2. When the user clicks Hang Up
  const endCall = () => {
    setToken("");
    setUrl("");
  };

  // 3. If connected, show the red "Hang Up" button and connect the mic!
  if (token && url) {
    return (
      <LiveKitRoom
        serverUrl={url}
        token={token}
        connect={true}
        audio={true} // Automatically asks for Microphone access
        video={false}
        onDisconnected={endCall}
      >
        <RoomAudioRenderer /> {/* This is the invisible speaker that plays the AI's voice */}
        <button
          onClick={endCall}
          className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-5 py-2.5 rounded-full font-bold transition-all shadow-[0_0_15px_rgba(239,68,68,0.5)] animate-pulse"
        >
          <PhoneOff className="w-5 h-5" /> Hang Up
        </button>
      </LiveKitRoom>
    );
  }

  // 4. Default state: Show the blue "Start Call" button
  return (
    <button
      onClick={startCall}
      disabled={isConnecting}
      className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-full font-bold transition-all shadow-lg hover:scale-105"
    >
      {isConnecting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Phone className="w-5 h-5" />}
      {isConnecting ? "Connecting..." : "Start Call"}
    </button>
  );
};

export default CallButton;