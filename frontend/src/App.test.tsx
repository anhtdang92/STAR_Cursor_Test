import { render, screen } from '@testing-library/react';
import App from './App';

describe('App Component', () => {
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
}); 