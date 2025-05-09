import React, { useEffect, useState } from 'react';
import { CircularProgress, Button } from '@mui/material';
import styled from 'styled-components';

interface ProcessingStatusProps {
  taskId: string;
  onComplete: (outputUrl: string) => void;
}

interface Progress {
  current_frame: number;
  total_frames: number;
  fps: number;
  percentage: number;
  status: string;
  estimated_time_remaining: number | null;
}

const Container = styled.div`
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

interface StatusIconProps {
  status: ProcessingStatusProps['status'];
}

const StatusIcon = styled.div<StatusIconProps>`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  background: ${(props: StatusIconProps) => {
    switch (props.status) {
      case 'processing':
        return '#007AFF';
      case 'completed':
        return '#34C759';
      case 'error':
        return '#FF3B30';
      default:
        return '#8E8E93';
    }
  }};
`;

const ProgressBar = styled.div`
  height: 4px;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 2px;
  overflow: hidden;
  margin: 16px 0;
`;

interface ProgressProps {
  progress: number;
}

const Progress = styled.div<ProgressProps>`
  height: 100%;
  width: ${(props: ProgressProps) => props.progress}%;
  background: #007AFF;
  border-radius: 2px;
  transition: width 0.3s ease;
`;

const Message = styled.p`
  margin: 0;
  color: #1D1D1F;
  font-size: 14px;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
`;

const Spinner = styled.div`
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ taskId, onComplete }) => {
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState(false);

  const formatTime = (seconds: number | null): string => {
    if (seconds === null) return 'Calculating...';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  useEffect(() => {
    const pollProgress = async () => {
      try {
        const response = await fetch(`/api/progress/${taskId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch progress');
        }
        
        const data = await response.json();
        setProgress(data);
        
        if (data.status === 'complete') {
          onComplete(data.output_url);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    const interval = setInterval(pollProgress, 1000);
    return () => clearInterval(interval);
  }, [taskId, onComplete]);

  const handleCancel = async () => {
    try {
      await fetch(`/api/cancel/${taskId}`, { method: 'POST' });
      setCancelled(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel processing');
    }
  };

  const handleDownloadPartial = async () => {
    try {
      const response = await fetch(`/api/partial/${taskId}`);
      if (!response.ok) {
        throw new Error('No partial video available');
      }
      
      // Create download link
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'partial_output.mp4';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download partial video');
    }
  };

  if (error) {
    return (
      <Container>
        <p className="text-red-600">{error}</p>
        {cancelled && (
          <Button
            variant="contained"
            color="primary"
            onClick={handleDownloadPartial}
            className="mt-2"
          >
            Download Partial Result
          </Button>
        )}
      </Container>
    );
  }

  if (!progress) {
    return <CircularProgress />;
  }

  return (
    <Container>
      <div className="flex items-center justify-between mb-2">
        <div className="flex-1">
          <div className="h-2 bg-gray-200 rounded">
            <div
              className="h-2 bg-blue-500 rounded"
              style={{ width: `${progress.percentage}%` }}
            />
          </div>
        </div>
        <span className="ml-4">{Math.round(progress.percentage)}%</span>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="text-sm text-gray-600">
          <div className="font-semibold">Frame Progress</div>
          <div>Frame {progress.current_frame.toLocaleString()} of {progress.total_frames.toLocaleString()}</div>
        </div>
        
        <div className="text-sm text-gray-600">
          <div className="font-semibold">Video Information</div>
          <div>FPS: {progress.fps.toFixed(2)}</div>
          <div>Time Remaining: {formatTime(progress.estimated_time_remaining)}</div>
        </div>
      </div>
      
      <div className="mt-4 flex justify-between">
        <Button
          variant="contained"
          color="secondary"
          onClick={handleCancel}
          disabled={cancelled}
        >
          {cancelled ? 'Cancelled' : 'Cancel'}
        </Button>
        
        {cancelled && (
          <Button
            variant="contained"
            color="primary"
            onClick={handleDownloadPartial}
          >
            Download Partial Result
          </Button>
        )}
      </div>
    </Container>
  );
};

export default ProcessingStatus; 