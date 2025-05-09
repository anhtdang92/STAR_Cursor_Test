import React, { useCallback, useState } from 'react';
import styled from 'styled-components';

interface VideoUploadProps {
  onUpload: (file: File) => void;
  className?: string;
}

const UploadContainer = styled.div`
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border: 2px dashed #007AFF;
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    background: rgba(0, 122, 255, 0.05);
    border-color: #0055FF;
  }
`;

const UploadIcon = styled.div`
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  color: #007AFF;
`;

const Title = styled.h3`
  margin: 0 0 8px;
  color: #1D1D1F;
  font-size: 20px;
  font-weight: 600;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
`;

const Subtitle = styled.p`
  margin: 0;
  color: #86868B;
  font-size: 14px;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
`;

const FileInput = styled.input`
  display: none;
`;

const ErrorMessage = styled.div`
  margin-top: 16px;
  color: #FF3B30;
  font-size: 14px;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
`;

export const VideoUpload: React.FC<VideoUploadProps> = ({ onUpload, className }) => {
  const [error, setError] = useState<string>('');

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== 'video/mp4') {
      setError('Please upload an MP4 video file');
      return;
    }

    if (file.size > 100 * 1024 * 1024) { // 100MB limit
      setError('File size must be less than 100MB');
      return;
    }

    setError('');
    onUpload(file);
  }, [onUpload]);

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    const file = event.dataTransfer.files?.[0];
    if (!file) return;

    if (file.type !== 'video/mp4') {
      setError('Please upload an MP4 video file');
      return;
    }

    if (file.size > 100 * 1024 * 1024) { // 100MB limit
      setError('File size must be less than 100MB');
      return;
    }

    setError('');
    onUpload(file);
  }, [onUpload]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
  }, []);

  return (
    <UploadContainer
      className={className}
      onClick={() => document.getElementById('fileInput')?.click()}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <UploadIcon>
        <svg width="64" height="64" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
        </svg>
      </UploadIcon>
      <Title>Upload Video</Title>
      <Subtitle>Drag and drop your MP4 video here, or click to browse</Subtitle>
      <FileInput
        id="fileInput"
        type="file"
        accept="video/mp4"
        onChange={handleFileChange}
      />
      {error && <ErrorMessage>{error}</ErrorMessage>}
    </UploadContainer>
  );
}; 