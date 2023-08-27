import React, { useState } from 'react';
import axios from 'axios';

const ChiSquareCalculator = () => {
  const [attribute1Data, setAttribute1Data] = useState('');
  const [attribute2Data, setAttribute2Data] = useState('');
  const [results, setResults] = useState(null);

  const handleCalculateChiSquare = () => {
    const postData = {
      attribute1_data: attribute1Data.split(',').map(value => value.trim()),
      attribute2_data: attribute2Data.split(',').map(value => value.trim())
    };
    console.log(postData)

    axios.post('http://127.0.0.1:8000/chi', postData)
      .then(response => {
        setResults(response.data);
      })
      .catch(error => {
        console.error('Error:', error);
      });
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">Chi-Square Calculator</h2>
      <div className="mb-4">
        <label className="block mb-2">Attribute 1 Data (comma-separated)</label>
        <textarea
          className="w-full p-2 border rounded"
          rows="3"
          value={attribute1Data}
          onChange={e => setAttribute1Data(e.target.value)}
        />
      </div>
      <div className="mb-4">
        <label className="block mb-2">Attribute 2 Data (comma-separated)</label>
        <textarea
          className="w-full p-2 border rounded"
          rows="3"
          value={attribute2Data}
          onChange={e => setAttribute2Data(e.target.value)}
        />
      </div>
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        onClick={handleCalculateChiSquare}
      >
        Calculate Chi-Square
      </button>
      {results && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Results</h3>
          <p>Chi-Square: {results.chi2}</p>
          <p>P-Value: {results.p_value}</p>
          <p>Degrees of Freedom: {results.degrees_of_freedom}</p>
          <h4 className="text-md font-semibold mt-2 mb-1">Contingency Table</h4>
          <table className="border-collapse border border-gray-400">
            <thead>
              <tr>
                <th className="border border-gray-400 p-2">Attribute 1</th>
                <th className="border border-gray-400 p-2">Attribute 2</th>
              </tr>
            </thead>
            <tbody>
              {results.contingency_table.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} className="border border-gray-400 p-2">{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default ChiSquareCalculator;
