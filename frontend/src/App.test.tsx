import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('App Component', () => {
  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
  });

  test('renders Video Upscaler title', () => {
    render(<App />);
    const titleElement = screen.getByText(/Video Upscaler/i);
    expect(titleElement).toBeInTheDocument();
  });

  test('renders upload area', () => {
    render(<App />);
    const uploadText = screen.getByText(/Drag and drop an MP4 video here/i);
    expect(uploadText).toBeInTheDocument();
  });

  test('handles file upload', async () => {
    // Mock successful upload response
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        task_id: 'test-task-id',
        status: 'processing'
      }
    });

    render(<App />);
    
    // Create a test file
    const file = new File(['test video content'], 'test.mp4', { type: 'video/mp4' });
    
    // Get the file input
    const fileInput = screen.getByTestId('file-input');
    
    // Upload the file
    await userEvent.upload(fileInput, file);
    
    // Check if upload was called with correct data
    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.stringContaining('/api/upload'),
      expect.any(FormData)
    );
    
    // Check if status is displayed
    await waitFor(() => {
      expect(screen.getByText(/Processing/i)).toBeInTheDocument();
    });
  });

  test('handles upload error', async () => {
    // Mock failed upload response
    mockedAxios.post.mockRejectedValueOnce(new Error('Upload failed'));

    render(<App />);
    
    // Create a test file
    const file = new File(['test video content'], 'test.mp4', { type: 'video/mp4' });
    
    // Get the file input
    const fileInput = screen.getByTestId('file-input');
    
    // Upload the file
    await userEvent.upload(fileInput, file);
    
    // Check if error message is displayed
    await waitFor(() => {
      expect(screen.getByText(/Error uploading file/i)).toBeInTheDocument();
    });
  });

  test('handles file type validation', async () => {
    render(<App />);
    
    // Create an invalid file
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    
    // Get the file input
    const fileInput = screen.getByTestId('file-input');
    
    // Try to upload the file
    await userEvent.upload(fileInput, file);
    
    // Check if error message is displayed
    await waitFor(() => {
      expect(screen.getByText(/Please upload an MP4 file/i)).toBeInTheDocument();
    });
  });

  test('updates status periodically', async () => {
    // Mock successful upload response
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        task_id: 'test-task-id',
        status: 'processing'
      }
    });

    // Mock status check responses
    mockedAxios.get
      .mockResolvedValueOnce({ data: { status: 'processing' } })
      .mockResolvedValueOnce({ data: { status: 'completed' } });

    render(<App />);
    
    // Create and upload a test file
    const file = new File(['test video content'], 'test.mp4', { type: 'video/mp4' });
    const fileInput = screen.getByTestId('file-input');
    await userEvent.upload(fileInput, file);
    
    // Check initial status
    await waitFor(() => {
      expect(screen.getByText(/Processing/i)).toBeInTheDocument();
    });
    
    // Wait for status update
    await waitFor(() => {
      expect(screen.getByText(/Completed/i)).toBeInTheDocument();
    }, { timeout: 3000 });
  });
}); 