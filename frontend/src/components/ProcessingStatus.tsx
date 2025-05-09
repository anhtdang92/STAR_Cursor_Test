import React from 'react';
import styled from 'styled-components';

interface ProcessingStatusProps {
  status: 'idle' | 'processing' | 'completed' | 'error';
  progress?: number;
  message?: string;
  className?: string;
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

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  status,
  progress = 0,
  message,
  className
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'processing':
        return <Spinner />;
      case 'completed':
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
          </svg>
        );
      case 'error':
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z" />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <Container className={className}>
      <StatusIcon status={status}>
        {getStatusIcon()}
      </StatusIcon>
      {status === 'processing' && (
        <ProgressBar>
          <Progress progress={progress} />
        </ProgressBar>
      )}
      {message && <Message>{message}</Message>}
    </Container>
  );
}; 