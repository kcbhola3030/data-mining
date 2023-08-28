import React, { useState } from 'react';

const Normalization = () => {
  const [selectedMethod, setSelectedMethod] = useState('');
  const [inputValues, setInputValues] = useState('');
  const [normalizedValues, setNormalizedValues] = useState('');

  const handleInputChange = (event) => {
    setInputValues(event.target.value);
  };

  const handleMethodChange = (event) => {
    setSelectedMethod(event.target.value);
  };

  const handleNormalize = () => {
    const values = inputValues.split(',').map(Number);
    let normalized = [];

    switch (selectedMethod) {
      case 'minmax':
        // Apply min-max normalization
        const min = Math.min(...values);
        const max = Math.max(...values);
        normalized = values.map((value) => (value - min) / (max - min));
        break;
      case 'zscore':
        // Apply z-score normalization
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
        const stdDev = Math.sqrt(values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length);
        normalized = values.map((value) => (value - mean) / stdDev);
        break;
      case 'decimalscaling':
        // Apply decimal scaling
        const maxMagnitude = Math.max(...values.map((value) => Math.floor(Math.log10(Math.abs(value))) + 1));
        normalized = values.map((value) => value / Math.pow(10, maxMagnitude));
        break;
      default:
        break;
    }

    setNormalizedValues(normalized);
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">Normalization</h2>
      <div className="mb-4">
        <label htmlFor="inputValues" className="block font-medium mb-1">
          Enter comma-separated values:
        </label>
        <input
          type="text"
          id="inputValues"
          className="border rounded p-2 w-full"
          value={inputValues}
          onChange={handleInputChange}
        />
      </div>
      <div className="mb-4">
        <label htmlFor="method" className="block font-medium mb-1">
          Select normalization method:
        </label>
        <select
          id="method"
          className="border rounded p-2 w-full"
          value={selectedMethod}
          onChange={handleMethodChange}
        >
          <option value="">Select method</option>
          <option value="minmax">Min-Max Normalization</option>
          <option value="zscore">Z-Score Normalization</option>
          <option value="decimalscaling">Decimal Scaling</option>
        </select>
      </div>
      <button
        onClick={handleNormalize}
        className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
      >
        Normalize
      </button>
      {normalizedValues.length > 0 && (
        <div className="mt-4">
          <h2 className="text-xl font-semibold">Normalized Values:</h2>
          <p>{normalizedValues.join(', ')}</p>
        </div>
      )}
    </div>
  );
};

export default Normalization;
