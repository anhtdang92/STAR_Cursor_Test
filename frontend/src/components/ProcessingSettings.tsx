import React from 'react';
import styled from 'styled-components';

export interface ProcessingSettings {
  upscaleFactor: number;
  model: 'artemis' | 'gaia' | 'theia';
  denoiseLevel: number;
  enhanceDetails: boolean;
}

interface ProcessingSettingsProps {
  settings: ProcessingSettings;
  onChange: (settings: ProcessingSettings) => void;
  className?: string;
}

const Container = styled.div`
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h3`
  margin: 0 0 24px;
  color: #1D1D1F;
  font-size: 20px;
  font-weight: 600;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
`;

const SettingGroup = styled.div`
  margin-bottom: 24px;

  &:last-child {
    margin-bottom: 0;
  }
`;

const SettingLabel = styled.label`
  display: block;
  margin-bottom: 8px;
  color: #1D1D1F;
  font-size: 14px;
  font-weight: 500;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
`;

const Select = styled.select`
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #D2D2D7;
  border-radius: 8px;
  background: white;
  color: #1D1D1F;
  font-size: 14px;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  appearance: none;
  cursor: pointer;

  &:focus {
    outline: none;
    border-color: #007AFF;
    box-shadow: 0 0 0 2px rgba(0, 122, 255, 0.2);
  }
`;

const Slider = styled.input`
  width: 100%;
  -webkit-appearance: none;
  height: 4px;
  background: #D2D2D7;
  border-radius: 2px;
  outline: none;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px;
    height: 20px;
    background: white;
    border: 1px solid #D2D2D7;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  &:focus::-webkit-slider-thumb {
    border-color: #007AFF;
    box-shadow: 0 0 0 2px rgba(0, 122, 255, 0.2);
  }
`;

const SliderValue = styled.div`
  margin-top: 8px;
  color: #86868B;
  font-size: 12px;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  text-align: right;
`;

const Toggle = styled.label`
  display: flex;
  align-items: center;
  cursor: pointer;
`;

const ToggleInput = styled.input`
  display: none;
`;

const ToggleSlider = styled.div<{ checked: boolean }>`
  position: relative;
  width: 50px;
  height: 30px;
  background: ${props => props.checked ? '#34C759' : '#D2D2D7'};
  border-radius: 15px;
  transition: background-color 0.2s ease;
  margin-right: 12px;

  &:before {
    content: '';
    position: absolute;
    top: 2px;
    left: ${props => props.checked ? '22px' : '2px'};
    width: 26px;
    height: 26px;
    background: white;
    border-radius: 50%;
    transition: left 0.2s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
`;

const ToggleLabel = styled.span`
  color: #1D1D1F;
  font-size: 14px;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
`;

export const ProcessingSettings: React.FC<ProcessingSettingsProps> = ({
  settings,
  onChange,
  className
}) => {
  const handleChange = (key: keyof ProcessingSettings, value: any) => {
    onChange({
      ...settings,
      [key]: value
    });
  };

  return (
    <Container className={className}>
      <Title>Processing Settings</Title>
      
      <SettingGroup>
        <SettingLabel>Upscale Factor</SettingLabel>
        <Select
          value={settings.upscaleFactor}
          onChange={(e) => handleChange('upscaleFactor', Number(e.target.value))}
        >
          <option value={2}>2x</option>
          <option value={4}>4x</option>
          <option value={8}>8x</option>
        </Select>
      </SettingGroup>

      <SettingGroup>
        <SettingLabel>AI Model</SettingLabel>
        <Select
          value={settings.model}
          onChange={(e) => handleChange('model', e.target.value)}
        >
          <option value="artemis">Artemis (Balanced)</option>
          <option value="gaia">Gaia (Quality)</option>
          <option value="theia">Theia (Speed)</option>
        </Select>
      </SettingGroup>

      <SettingGroup>
        <SettingLabel>Denoise Level</SettingLabel>
        <Slider
          type="range"
          min="0"
          max="100"
          value={settings.denoiseLevel}
          onChange={(e) => handleChange('denoiseLevel', Number(e.target.value))}
        />
        <SliderValue>{settings.denoiseLevel}%</SliderValue>
      </SettingGroup>

      <SettingGroup>
        <Toggle>
          <ToggleInput
            type="checkbox"
            checked={settings.enhanceDetails}
            onChange={(e) => handleChange('enhanceDetails', e.target.checked)}
          />
          <ToggleSlider checked={settings.enhanceDetails} />
          <ToggleLabel>Enhance Details</ToggleLabel>
        </Toggle>
      </SettingGroup>
    </Container>
  );
}; 