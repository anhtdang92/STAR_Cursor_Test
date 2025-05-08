import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

interface UploadStatus {
  status: 'idle' | 'uploading' | 'processing' | 'complete' | 'error';
  progress: number;
  error?: string;
  downloadUrl?: string;
}

const App: React.FC = () => {
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({
    status: 'idle',
    progress: 0
  });

  const onDrop = async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (!file.type.includes('video/mp4')) {
      setUploadStatus({
        status: 'error',
        progress: 0,
        error: 'Please upload an MP4 file'
      });
      return;
    }

    const formData = new FormData();
    formData.append('video', file);

    try {
      setUploadStatus({ status: 'uploading', progress: 0 });

      // Upload video
      const uploadResponse = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.loaded / (progressEvent.total || 1) * 100;
          setUploadStatus(prev => ({ ...prev, progress }));
        }
      });

      const { taskId } = uploadResponse.data;
      setUploadStatus({ status: 'processing', progress: 50 });

      // Poll for status
      const checkStatus = async () => {
        const statusResponse = await axios.get(`http://localhost:5000/api/status/${taskId}`);
        const { status, downloadUrl } = statusResponse.data;

        if (status === 'complete') {
          setUploadStatus({
            status: 'complete',
            progress: 100,
            downloadUrl
          });
        } else if (status === 'error') {
          setUploadStatus({
            status: 'error',
            progress: 0,
            error: 'Processing failed'
          });
        } else {
          setTimeout(checkStatus, 2000);
        }
      };

      checkStatus();
    } catch (error) {
      setUploadStatus({
        status: 'error',
        progress: 0,
        error: 'Upload failed'
      });
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/mp4': ['.mp4']
    },
    maxFiles: 1
  });

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">
            Video Upscaler
          </h1>
        </div>

        <div className="bg-white shadow sm:rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
                ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}
            >
              <input {...getInputProps()} />
              {uploadStatus.status === 'idle' && (
                <div>
                  <p className="text-gray-600">
                    Drag and drop an MP4 video here, or click to select
                  </p>
                </div>
              )}
            </div>

            {uploadStatus.status !== 'idle' && (
              <div className="mt-6">
                <div className="relative pt-1">
                  <div className="flex mb-2 items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-blue-600 bg-blue-200">
                        {uploadStatus.status}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs font-semibold inline-block text-blue-600">
                        {Math.round(uploadStatus.progress)}%
                      </span>
                    </div>
                  </div>
                  <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-blue-200">
                    <div
                      style={{ width: `${uploadStatus.progress}%` }}
                      className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
                    />
                  </div>
                </div>

                {uploadStatus.error && (
                  <div className="mt-4 text-red-600">
                    {uploadStatus.error}
                  </div>
                )}

                {uploadStatus.downloadUrl && (
                  <div className="mt-4">
                    <a
                      href={uploadStatus.downloadUrl}
                      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Download Upscaled Video
                    </a>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App; 