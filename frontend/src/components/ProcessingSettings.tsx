import React from 'react';
import styled from 'styled-components';

export interface ProcessingSettings {
  upscaleFactor: number;
  model: string;
  denoiseLevel: number;
  enhanceDetails: boolean;
}

interface ProcessingSettingsProps {
  settings: ProcessingSettings;
  onChange: (settings: ProcessingSettings) => void;
}

const Container = styled.div`
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h2`
  margin: 0 0 20px;
  color: #1D1D1F;
  font-size: 20px;
  font-weight: 600;
`;

const SettingGroup = styled.div`
  margin-bottom: 20px;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 8px;
  color: #1D1D1F;
  font-size: 14px;
  font-weight: 500;
`;

const Select = styled.select`
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #D1D1D6;
  border-radius: 6px;
  font-size: 14px;
  color: #1D1D1F;
  background: white;
  
  &:focus {
    outline: none;
    border-color: #007AFF;
  }
`;

const RangeInput = styled.input`
  width: 100%;
  margin: 8px 0;
`;

const Checkbox = styled.input`
  margin-right: 8px;
`;

const ProcessingSettingsComponent: React.FC<ProcessingSettingsProps> = ({ settings, onChange }) => {
  const handleChange = (key: keyof ProcessingSettings, value: any) => {
    onChange({
      ...settings,
      [key]: value
    });
  };

  return (
    <Container>
      <Title>Processing Settings</Title>
      
      <SettingGroup>
        <Label>Upscale Factor</Label>
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
        <Label>Model</Label>
        <Select
          value={settings.model}
          onChange={(e) => handleChange('model', e.target.value)}
        >
          <option value="artemis">Artemis</option>
          <option value="gaia">Gaia</option>
          <option value="theia">Theia</option>
        </Select>
      </SettingGroup>
      
      <SettingGroup>
        <Label>Denoise Level: {settings.denoiseLevel}%</Label>
        <RangeInput
          type="range"
          min="0"
          max="100"
          value={settings.denoiseLevel}
          onChange={(e) => handleChange('denoiseLevel', Number(e.target.value))}
        />
      </SettingGroup>
      
      <SettingGroup>
        <Label>
          <Checkbox
            type="checkbox"
            checked={settings.enhanceDetails}
            onChange={(e) => handleChange('enhanceDetails', e.target.checked)}
          />
          Enhance Details
        </Label>
      </SettingGroup>
    </Container>
  );
};

export default ProcessingSettingsComponent; 