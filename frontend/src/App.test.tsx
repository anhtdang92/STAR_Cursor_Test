import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import axios from 'axios';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('App Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockedAxios.post.mockResolvedValue({ data: { taskId: '123' } });
    mockedAxios.get.mockResolvedValue({ data: { status: 'complete', downloadUrl: 'http://example.com/video.mp4' } });
  });

  test('renders title and introduction', () => {
    render(<App />);
    expect(screen.getByText('STAR Video Upscaler')).toBeInTheDocument();
    expect(screen.getByText(/Enhance Your Videos with AI/)).toBeInTheDocument();
  });

  test('handles file upload successfully', async () => {
    render(<App />);
    const file = new File(['video content'], 'test.mp4', { type: 'video/mp4' });
    const input = screen.getByTestId('file-input');
    
    await userEvent.upload(input, file);

    // Check if the upload API was called
    expect(mockedAxios.post).toHaveBeenCalledWith(
      'http://localhost:5000/api/upload',
      expect.any(FormData),
      expect.any(Object)
    );

    // Check if the status API was called
    await waitFor(() => {
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'http://localhost:5000/api/status/123'
      );
    });

    // Wait for the download button to appear
    await waitFor(() => {
      const downloadButton = screen.getByText(/Download Enhanced Video/);
      expect(downloadButton).toBeInTheDocument();
      expect(downloadButton.closest('a')).toHaveAttribute('href', 'http://example.com/video.mp4');
    });
  });

  test('shows error for non-MP4 files', async () => {
    render(<App />);
    const file = new File(['content'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByTestId('file-input');
    
    await userEvent.upload(input, file);
    
    // Check for error message
    await waitFor(() => {
      expect(screen.getByText('Please upload an MP4 file')).toBeInTheDocument();
    });
  });

  test('shows processing error message', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { status: 'error', error: 'Processing failed' } });
    
    render(<App />);
    const file = new File(['video content'], 'test.mp4', { type: 'video/mp4' });
    const input = screen.getByTestId('file-input');
    
    await userEvent.upload(input, file);
    
    await waitFor(() => {
      expect(screen.getByText('Processing failed')).toBeInTheDocument();
    });
  });

  test('allows resetting after error', async () => {
    render(<App />);
    const file = new File(['content'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByTestId('file-input');
    
    await userEvent.upload(input, file);
    
    await waitFor(() => {
      expect(screen.getByText('Please upload an MP4 file')).toBeInTheDocument();
    });
    
    const tryAgainButton = screen.getByText('Try Again');
    await userEvent.click(tryAgainButton);
    
    expect(screen.queryByText('Please upload an MP4 file')).not.toBeInTheDocument();
  });

  test('allows processing another video after successful upload', async () => {
    render(<App />);
    const file = new File(['video content'], 'test.mp4', { type: 'video/mp4' });
    const input = screen.getByTestId('file-input');
    
    await userEvent.upload(input, file);
    
    await waitFor(() => {
      expect(screen.getByText(/Download Enhanced Video/)).toBeInTheDocument();
    });
    
    const processAnotherButton = screen.getByText('Process Another Video');
    await userEvent.click(processAnotherButton);
    
    expect(screen.queryByText(/Download Enhanced Video/)).not.toBeInTheDocument();
  });
}); 