import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import Button from './Button';
import Card from './Card';
import ProgressBar from './ProgressBar';
import type { UpscaleSettings } from './UpscaleSettings';

interface VideoUploaderProps {
  onUploadComplete: (downloadUrl: string) => void;
  onError: (error: string) => void;
  settings: UpscaleSettings;
}

const VideoUploader: React.FC<VideoUploaderProps> = ({
  onUploadComplete,
  onError,
  settings
}) => {
  const [uploadStatus, setUploadStatus] = useState<{
    status: 'idle' | 'uploading' | 'processing' | 'complete' | 'error';
    progress: number;
    error?: string;
  }>({
    status: 'idle',
    progress: 0
  });

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (!file.type.includes('video/mp4')) {
      onError('Please upload an MP4 file');
      return;
    }

    const formData = new FormData();
    formData.append('video', file);
    formData.append('settings', JSON.stringify(settings));

    try {
      setUploadStatus({ status: 'uploading', progress: 0 });

      const uploadResponse = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.loaded / (progressEvent.total || 1) * 100;
          setUploadStatus(prev => ({ ...prev, progress }));
        }
      });

      const { taskId } = uploadResponse.data;
      setUploadStatus(prev => ({ ...prev, status: 'processing', progress: 50 }));

      // Poll for status
      const checkStatus = async () => {
        const statusResponse = await axios.get(`http://localhost:5000/api/status/${taskId}`);
        const { status, downloadUrl, error, progress } = statusResponse.data;

        if (status === 'complete' && downloadUrl) {
          setUploadStatus({ status: 'complete', progress: 100 });
          onUploadComplete(downloadUrl);
        } else if (status === 'error' || error) {
          setUploadStatus({
            status: 'error',
            progress: 0,
            error: error || 'Processing failed'
          });
          onError(error || 'Processing failed');
        } else {
          setUploadStatus(prev => ({
            ...prev,
            progress: 50 + (progress || 0) * 0.5 // Map processing progress to 50-100%
          }));
          setTimeout(checkStatus, 2000);
        }
      };

      checkStatus();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setUploadStatus({
        status: 'error',
        progress: 0,
        error: errorMessage
      });
      onError(errorMessage);
    }
  }, [onUploadComplete, onError, settings]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/mp4': ['.mp4']
    },
    maxFiles: 1,
    disabled: uploadStatus.status === 'uploading' || uploadStatus.status === 'processing'
  });

  const getStatusText = () => {
    switch (uploadStatus.status) {
      case 'uploading':
        return 'Uploading...';
      case 'processing':
        return `Processing with STAR AI (${settings.scale}x upscale, ${settings.quality} quality)...`;
      case 'complete':
        return 'Processing complete!';
      case 'error':
        return uploadStatus.error || 'Error occurred';
      default:
        return isDragActive
          ? 'Drop the MP4 file here'
          : 'Drag and drop an MP4 video here, or click to select';
    }
  };

  return (
    <Card variant="elevated" className="p-6">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-apple p-8 text-center transition-all duration-200
          ${isDragActive ? 'border-apple-blue-500 bg-apple-blue-500/5' : 'border-apple-gray-200'}
          ${uploadStatus.status === 'uploading' || uploadStatus.status === 'processing'
            ? 'cursor-not-allowed opacity-50'
            : 'cursor-pointer hover:border-apple-blue-500 hover:bg-apple-blue-500/5'
          }
        `}
      >
        <input {...getInputProps()} data-testid="file-input" />
        <p className="text-apple-gray-500 text-lg mb-2">{getStatusText()}</p>
        {(uploadStatus.status === 'uploading' || uploadStatus.status === 'processing') && (
          <div className="mt-4">
            <ProgressBar
              progress={uploadStatus.progress}
              size="large"
            />
          </div>
        )}
      </div>

      {/* Settings Summary */}
      {uploadStatus.status === 'idle' && (
        <div className="mt-4 p-4 bg-apple-gray-50 rounded-lg">
          <p className="text-sm text-apple-gray-400">
            Selected settings: {settings.scale}x upscale, {settings.quality} quality
            {settings.denoiseLevel > 0 && `, denoise level ${settings.denoiseLevel}`}
            {settings.preserveDetails && ', preserve details'}
          </p>
        </div>
      )}
    </Card>
  );
};

export default VideoUploader; 