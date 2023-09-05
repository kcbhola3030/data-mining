// src/Classifier.js

import React, { useState } from 'react';
import axios from 'axios';

function Classifier() {
  const [selectedMethod, setSelectedMethod] = useState('info_gain');
  const [file, setFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [results, setResults] = useState(null);

  const handleMethodChange = (event) => {
    setSelectedMethod(event.target.value);
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleTargetColumnChange = (event) => {
    setTargetColumn(event.target.value);
  };

  const handleSubmit = async () => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_column', targetColumn);

      const response = await axios.post(`http://127.0.0.1:8000/classify/${selectedMethod}/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <h1>Classifier Page</h1>
      <label>
        Select Method:
        <select value={selectedMethod} onChange={handleMethodChange}>
          <option value="info_gain">Information Gain</option>
          <option value="gain_ratio">Gain Ratio</option>
          <option value="gini">Gini Index</option>
        </select>
      </label>
      <br />
      <label>
        Upload Dataset:
        <input type="file" accept=".csv" onChange={handleFileChange} />
      </label>
      <br />
      <label>
        Target Column:
        <input type="text" value={targetColumn} onChange={handleTargetColumnChange} />
      </label>
      <br />
      <button onClick={handleSubmit}>Submit</button>
      {results && (
        <div>
          <h2>Classification Results:</h2>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default Classifier;
