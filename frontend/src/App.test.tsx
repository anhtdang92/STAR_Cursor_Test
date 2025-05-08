import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import { act } from 'react';
import axios from 'axios';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('App Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockedAxios.post.mockResolvedValue({ data: { taskId: '123' } });
    mockedAxios.get.mockResolvedValue({ data: { status: 'complete', downloadUrl: 'http://example.com/video.mp4' } });
  });

  test('renders title and upload area', () => {
    render(<App />);
    expect(screen.getByText('Video Upscaler')).toBeInTheDocument();
    expect(screen.getByText('Drag and drop an MP4 video here, or click to select')).toBeInTheDocument();
  });

  test('handles file upload successfully', async () => {
    render(<App />);
    const file = new File(['video content'], 'test.mp4', { type: 'video/mp4' });
    const input = screen.getByTestId('file-input');
    
    await act(async () => {
      await userEvent.upload(input, file);
    });

    // Check if the upload API was called
    expect(mockedAxios.post).toHaveBeenCalled();

    // Check if the status API was called
    expect(mockedAxios.get).toHaveBeenCalled();

    // Wait for the download button to appear
    await waitFor(() => {
      const downloadButton = screen.getByRole('link', { name: /download upscaled video/i });
      expect(downloadButton).toBeInTheDocument();
      expect(downloadButton).toHaveAttribute('href', 'http://example.com/video.mp4');
    });
  });

  test('shows error for non-MP4 files', async () => {
    render(<App />);
    const file = new File(['content'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByTestId('file-input');
    
    await act(async () => {
      await userEvent.upload(input, file);
    });
    
    // Check for error message
    await waitFor(() => {
      const errorElement = screen.getByText('Please upload an MP4 file', { exact: true });
      expect(errorElement).toBeInTheDocument();
      expect(errorElement).toHaveClass('text-red-600');
    });
  });
}); 