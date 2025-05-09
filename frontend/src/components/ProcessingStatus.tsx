import React from 'react';

interface ProcessingStatusProps {
  status: string;
  progress: number;
  error?: string;
}

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ status, progress, error }) => {
  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">Processing Status</span>
        <span className="text-sm text-gray-500">{status}</span>
      </div>
      
      {status === 'processing' && (
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      )}
      
      {error && (
        <div className="mt-2 text-sm text-red-600">
          {error}
        </div>
      )}
    </div>
  );
};

export default ProcessingStatus; 