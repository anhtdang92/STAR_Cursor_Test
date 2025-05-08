import React from 'react';

interface ProgressBarProps {
  progress: number;
  showPercentage?: boolean;
  size?: 'small' | 'medium' | 'large';
  className?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  showPercentage = true,
  size = 'medium',
  className = ''
}) => {
  const normalizedProgress = Math.min(Math.max(progress, 0), 100);
  
  const sizes = {
    small: 'h-1',
    medium: 'h-2',
    large: 'h-3'
  };

  return (
    <div className={`w-full ${className}`}>
      <div className="flex items-center justify-between mb-1">
        {showPercentage && (
          <span className="text-sm font-medium text-apple-gray-400">
            {Math.round(normalizedProgress)}%
          </span>
        )}
      </div>
      <div className={`w-full bg-apple-gray-100 rounded-full overflow-hidden ${sizes[size]}`}>
        <div
          className="bg-apple-blue-500 transition-all duration-300 ease-in-out rounded-full"
          style={{ width: `${normalizedProgress}%` }}
        />
      </div>
    </div>
  );
};

export default ProgressBar; 