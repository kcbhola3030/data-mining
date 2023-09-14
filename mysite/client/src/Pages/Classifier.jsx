// src/Classifier.js

import React, { useState } from 'react';
import axios from 'axios';
// import decisionTreeImage from '../../../media/decision_tree.png';
import tree from '../../src/decision_tree.png'


// import a from '/Users/krishnacharanbhola/Desktop/MAMBA/sem7/DM/assign1/mysite/media/decision_tree.png'
function Classifier() {
  const [selectedMethod, setSelectedMethod] = useState('info_gain');
  const [file, setFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [results, setResults] = useState(null);
  const [imageURL, setImageURL] = useState('');

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
      
      console.log(response)
      setImageURL(response?.data.image_url);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-semibold mb-4">Classifier Page</h1>
      <label className="block mb-2">
        Select Method:
        <select
          value={selectedMethod}
          onChange={handleMethodChange}
          className="w-full border rounded py-2 px-3"
        >
          <option value="info_gain">Information Gain</option>
          <option value="gain_ratio">Gain Ratio</option>
          <option value="gini">Gini Index</option>
        </select>
      </label>
      <br />
      <label className="block mb-2">
        Upload Dataset:
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="w-full border rounded py-2 px-3"
        />
      </label>
      <br />
      <label className="block mb-2">
        Target Column:
        <input
          type="text"
          value={targetColumn}
          onChange={handleTargetColumnChange}
          className="w-full border rounded py-2 px-3"
        />
      </label>
      <br />
      <button
        onClick={handleSubmit}
        className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
      >
        Submit
      </button>
      {results && (
        <div className="mt-4">
          <h2 className="text-lg font-semibold mb-2">Classification Results:</h2>
          <pre className="bg-gray-100 p-4 rounded">{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
      {imageURL && (
        <div>
          <h2>Decision Tree:</h2>
          <img src={tree} alt="Decision Tree" />
        </div>
      )}
    </div>
  );
}

export default Classifier;
