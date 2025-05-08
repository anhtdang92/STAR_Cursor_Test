import React from 'react';
import Card from './Card';

export interface UpscaleSettings {
  scale: 2 | 4;
  quality: 'fast' | 'balanced' | 'quality';
  denoiseLevel: 0 | 1 | 2 | 3;
  preserveDetails: boolean;
}

interface UpscaleSettingsProps {
  settings: UpscaleSettings;
  onSettingsChange: (settings: UpscaleSettings) => void;
}

const UpscaleSettings: React.FC<UpscaleSettingsProps> = ({
  settings,
  onSettingsChange
}) => {
  const handleChange = (key: keyof UpscaleSettings, value: any) => {
    onSettingsChange({
      ...settings,
      [key]: value
    });
  };

  return (
    <Card variant="default" className="p-6">
      <h3 className="text-xl font-semibold text-apple-gray-500 mb-6">
        Upscaling Settings
      </h3>
      
      <div className="space-y-6">
        {/* Scale Factor */}
        <div>
          <label className="block text-sm font-medium text-apple-gray-400 mb-2">
            Scale Factor
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all
                ${settings.scale === 2
                  ? 'bg-apple-blue-500 text-white'
                  : 'bg-apple-gray-100 text-apple-gray-500 hover:bg-apple-gray-200'
                }`}
              onClick={() => handleChange('scale', 2)}
            >
              2x
            </button>
            <button
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all
                ${settings.scale === 4
                  ? 'bg-apple-blue-500 text-white'
                  : 'bg-apple-gray-100 text-apple-gray-500 hover:bg-apple-gray-200'
                }`}
              onClick={() => handleChange('scale', 4)}
            >
              4x
            </button>
          </div>
        </div>

        {/* Quality Preset */}
        <div>
          <label className="block text-sm font-medium text-apple-gray-400 mb-2">
            Quality Preset
          </label>
          <div className="grid grid-cols-3 gap-2">
            {(['fast', 'balanced', 'quality'] as const).map((preset) => (
              <button
                key={preset}
                className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-all
                  ${settings.quality === preset
                    ? 'bg-apple-blue-500 text-white'
                    : 'bg-apple-gray-100 text-apple-gray-500 hover:bg-apple-gray-200'
                  }`}
                onClick={() => handleChange('quality', preset)}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        {/* Denoise Level */}
        <div>
          <label className="block text-sm font-medium text-apple-gray-400 mb-2">
            Denoise Level
          </label>
          <div className="grid grid-cols-4 gap-2">
            {([0, 1, 2, 3] as const).map((level) => (
              <button
                key={level}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all
                  ${settings.denoiseLevel === level
                    ? 'bg-apple-blue-500 text-white'
                    : 'bg-apple-gray-100 text-apple-gray-500 hover:bg-apple-gray-200'
                  }`}
                onClick={() => handleChange('denoiseLevel', level)}
              >
                {level}
              </button>
            ))}
          </div>
        </div>

        {/* Preserve Details */}
        <div className="flex items-center">
          <input
            type="checkbox"
            id="preserveDetails"
            className="h-4 w-4 text-apple-blue-500 focus:ring-apple-blue-500 border-apple-gray-200 rounded"
            checked={settings.preserveDetails}
            onChange={(e) => handleChange('preserveDetails', e.target.checked)}
          />
          <label
            htmlFor="preserveDetails"
            className="ml-2 block text-sm text-apple-gray-500"
          >
            Preserve fine details and textures
          </label>
        </div>

        {/* Settings Description */}
        <div className="mt-4 p-4 bg-apple-gray-50 rounded-lg">
          <p className="text-sm text-apple-gray-400">
            {settings.quality === 'fast' && (
              "Fast mode prioritizes speed while maintaining good quality. Best for quick previews and less complex videos."
            )}
            {settings.quality === 'balanced' && (
              "Balanced mode offers a good trade-off between processing speed and output quality. Recommended for most videos."
            )}
            {settings.quality === 'quality' && (
              "Quality mode uses advanced processing to achieve the best possible results. May take longer to process."
            )}
          </p>
        </div>
      </div>
    </Card>
  );
};

export default UpscaleSettings; 