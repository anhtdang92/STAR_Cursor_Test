import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'text';
  size?: 'small' | 'medium' | 'large';
  isLoading?: boolean;
  className?: string;
  href?: string;
  download?: boolean;
  onClick?: () => void;
  disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'medium',
  isLoading = false,
  className = '',
  href,
  download,
  onClick,
  disabled,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium transition-all duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variants = {
    primary: 'bg-apple-blue-500 text-white hover:bg-apple-blue-600 active:bg-apple-blue-700 focus:ring-apple-blue-500',
    secondary: 'bg-apple-gray-100 text-apple-gray-500 hover:bg-apple-gray-200 active:bg-apple-gray-300 focus:ring-apple-gray-200',
    text: 'text-apple-blue-500 hover:text-apple-blue-600 focus:ring-apple-blue-500'
  };

  const sizes = {
    small: 'px-3 py-1.5 text-sm rounded-lg',
    medium: 'px-4 py-2 text-base rounded-xl',
    large: 'px-6 py-3 text-lg rounded-2xl'
  };

  const loadingSpinner = (
    <svg className="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
  );

  const isDisabled = disabled || isLoading;
  const combinedClassName = `${baseStyles} ${variants[variant]} ${sizes[size]} ${className} ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}`;

  if (href) {
    return (
      <a
        href={href}
        className={combinedClassName}
        download={download}
        onClick={isDisabled ? (e: React.MouseEvent) => e.preventDefault() : onClick}
        {...props}
      >
        {isLoading && loadingSpinner}
        {children}
      </a>
    );
  }

  return (
    <button
      type="button"
      className={combinedClassName}
      disabled={isDisabled}
      onClick={onClick}
      {...props}
    >
      {isLoading && loadingSpinner}
      {children}
    </button>
  );
};

export default Button; 