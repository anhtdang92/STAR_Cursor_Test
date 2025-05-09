import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { VideoUpload } from './components/VideoUpload';
import { VideoPlayer } from './components/VideoPlayer';
import { ProcessingStatus } from './components/ProcessingStatus';
import { ProcessingSettings, ProcessingSettings as ProcessingSettingsType } from './components/ProcessingSettings';
import axios from 'axios';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #F5F5F7 0%, #FFFFFF 100%);
  padding: 40px;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
`;

const Content = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  align-items: start;
`;

const LeftColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 40px;
`;

const RightColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 40px;
`;

const Title = styled.h1`
  margin: 0 0 40px;
  color: #1D1D1F;
  font-size: 32px;
  font-weight: 700;
  text-align: center;
`;

const initialSettings: ProcessingSettingsType = {
  upscaleFactor: 2,
  model: 'artemis',
  denoiseLevel: 50,
  enhanceDetails: true,
};

function App() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'completed' | 'error'>('idle');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [settings, setSettings] = useState<ProcessingSettingsType>(initialSettings);

  const handleUpload = useCallback(async (file: File) => {
    setVideoFile(file);
    setVideoUrl(URL.createObjectURL(file));
    setProcessingStatus('idle');
    setProcessingProgress(0);
  }, []);

  const handleProcess = useCallback(async () => {
    if (!videoFile) return;

    setProcessingStatus('processing');
    setProcessingProgress(0);

    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('settings', JSON.stringify(settings));

    try {
      const response = await axios.post('http://localhost:5000/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.loaded / (progressEvent.total || 1) * 100;
          setProcessingProgress(progress);
        },
      });

      if (response.data.success) {
        setProcessingStatus('completed');
        // Handle the processed video URL
        setVideoUrl(response.data.processedVideoUrl);
      } else {
        setProcessingStatus('error');
      }
    } catch (error) {
      console.error('Error processing video:', error);
      setProcessingStatus('error');
    }
  }, [videoFile, settings]);

  return (
    <AppContainer>
      <Title>Video Upscaler</Title>
      <Content>
        <LeftColumn>
          <VideoUpload onUpload={handleUpload} />
          {videoUrl && (
            <VideoPlayer src={videoUrl} />
          )}
        </LeftColumn>
        <RightColumn>
          <ProcessingSettings
            settings={settings}
            onChange={setSettings}
          />
          <ProcessingStatus
            status={processingStatus}
            progress={processingProgress}
            message={
              processingStatus === 'processing' ? 'Processing your video...' :
              processingStatus === 'completed' ? 'Processing completed!' :
              processingStatus === 'error' ? 'An error occurred during processing.' :
              'Ready to process'
            }
          />
        </RightColumn>
      </Content>
    </AppContainer>
  );
}

export default App; 