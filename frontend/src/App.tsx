import React, { useState } from 'react';
import VideoUploader from './components/VideoUploader';
import Button from './components/Button';
import Card from './components/Card';

const App: React.FC = () => {
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUploadComplete = (url: string) => {
    setDownloadUrl(url);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setDownloadUrl(null);
  };

  const handleReset = () => {
    setDownloadUrl(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-apple-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-semibold text-apple-gray-500 text-center">
            STAR Video Upscaler
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="space-y-8">
          {/* Introduction */}
          <Card variant="default" className="p-6">
            <h2 className="text-xl font-semibold text-apple-gray-500 mb-4">
              Enhance Your Videos with AI
            </h2>
            <p className="text-apple-gray-400">
              Upload your MP4 video and our advanced AI model will upscale it to higher quality.
              The STAR model uses state-of-the-art technology to enhance video resolution while
              preserving details and reducing artifacts.
            </p>
          </Card>

          {/* Video Uploader */}
          <VideoUploader
            onUploadComplete={handleUploadComplete}
            onError={handleError}
          />

          {/* Error Display */}
          {error && (
            <div className="rounded-apple bg-red-50 p-4 border border-red-100">
              <p className="text-red-600">{error}</p>
              <Button
                variant="text"
                size="small"
                className="mt-2"
                onClick={handleReset}
              >
                Try Again
              </Button>
            </div>
          )}

          {/* Download Section */}
          {downloadUrl && (
            <Card variant="elevated" className="p-6 text-center">
              <h3 className="text-xl font-semibold text-apple-gray-500 mb-4">
                Your Video is Ready!
              </h3>
              <div className="space-x-4">
                <Button
                  href={downloadUrl}
                  download
                  variant="primary"
                  size="large"
                >
                  Download Enhanced Video
                </Button>
                <Button
                  variant="secondary"
                  size="large"
                  onClick={handleReset}
                >
                  Process Another Video
                </Button>
              </div>
            </Card>
          )}

          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card variant="default" className="p-6">
              <h3 className="text-lg font-semibold text-apple-gray-500 mb-2">
                High Quality
              </h3>
              <p className="text-apple-gray-400">
                Advanced AI upscaling for crystal-clear results
              </p>
            </Card>
            <Card variant="default" className="p-6">
              <h3 className="text-lg font-semibold text-apple-gray-500 mb-2">
                Fast Processing
              </h3>
              <p className="text-apple-gray-400">
                Optimized for speed without compromising quality
              </p>
            </Card>
            <Card variant="default" className="p-6">
              <h3 className="text-lg font-semibold text-apple-gray-500 mb-2">
                Easy to Use
              </h3>
              <p className="text-apple-gray-400">
                Simple drag-and-drop interface for quick results
              </p>
            </Card>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-apple-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-apple-gray-400">
            Powered by STAR AI Technology
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App; 