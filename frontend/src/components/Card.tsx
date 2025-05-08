import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'elevated';
}

const Card: React.FC<CardProps> = ({
  children,
  className = '',
  variant = 'default'
}) => {
  const baseStyles = 'bg-white rounded-apple transition-all duration-200';
  
  const variants = {
    default: 'border border-apple-gray-200',
    elevated: 'shadow-apple hover:shadow-apple-hover'
  };

  return (
    <div className={`${baseStyles} ${variants[variant]} ${className}`}>
      {children}
    </div>
  );
};

export default Card; 