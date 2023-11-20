import React, { useState } from 'react';
import axios from 'axios';

function Task1() {
  const [countOfSupportAndConfidence, setCountOfSupportAndConfidence] = useState(1);
  const [supportAndConfidence, setSupportAndConfidence] = useState([
    { support: '0.11', confidence: '0.51' },
    { support: '0.12', confidence: '0.52' },
    { support: '0.13', confidence: '0.52' }
  ]);
  const [results, setResults] = useState([]);
  const [error, setError] = useState();

  const runAssociationRules = async () => {
    try {
      const response = await axios.post('http://localhost:8000/run_association_rules/', {
        support_values: supportAndConfidence.map(item => item.support),
        confidence_values: supportAndConfidence.map(item => item.confidence),
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log('Response from the server:', response.data);

      if (Array.isArray(response.data)) {
        setResults(response.data);
        setError(null);  // Clear any previous errors
      } else {
        setError('Invalid response format');
      }
    } catch (error) {
      console.error('Error:', error.response?.data?.error || 'Unknown error');
      setError('Error occurred while fetching results');
    }
  };

  const handleCountChange = (e) => {
    if(e.target.value!=0)
    {
    
    const count = parseInt(e.target.value, 10);

      setCountOfSupportAndConfidence(count);
      
      // Reset the input fields to the default values with a difference of 0.01
      const newSupportAndConfidence = Array.from({ length: count }, (_, index) => ({
        support: (0.1 + index * 0.01).toFixed(2),
        confidence: (0.5 + index * 0.02).toFixed(2),
      }));
      setSupportAndConfidence(newSupportAndConfidence);
    }
  };
  

  const handleInputChange = (index, type, value) => {
    const newSupportAndConfidence = [...supportAndConfidence];
    newSupportAndConfidence[index] = { ...newSupportAndConfidence[index], [type]: value };
    setSupportAndConfidence(newSupportAndConfidence);
  };

  return (
    <div className="container">
      <h1 className="text-center text-3xl mb-5">Association Rule Mining</h1>
      <div className="mb-5">
        <label className="block mb-2 text-lg">Count of Support and Confidence:</label>
        <input
          type="number"
          value={countOfSupportAndConfidence}
          onChange={handleCountChange}
          className="w-full py-2 px-4 border border-gray-300 rounded-md focus:outline-none focus:border-blue-500"
        />
      </div>
      {[...Array(countOfSupportAndConfidence)].map((_, index) => (
        <div key={index} className="input-row mb-3 flex items-center">
          <label className="text-lg">Support Value {index + 1}:</label>
          <input
            type="number"
            step="0.01"
            value={supportAndConfidence[index].support}
            onChange={(e) => handleInputChange(index, 'support', e.target.value)}
            className="w-1/2 py-2 px-4 border border-gray-300 rounded-md focus:outline-none focus:border-blue-500 mr-3"
          />
          <label className="text-lg">Confidence Value {index + 1}:</label>
          <input
            type="number"
            step="0.01"
            value={supportAndConfidence[index].confidence}
            onChange={(e) => handleInputChange(index, 'confidence', e.target.value)}
            className="w-1/2 py-2 px-4 border border-gray-300 rounded-md focus:outline-none focus:border-blue-500"
          />
        </div>
      ))}
      <button
        onClick={runAssociationRules}
        className="block w-full py-3 px-4 bg-blue-500 text-white rounded-md hover:bg-blue-600"
      >
        Run Association Rules
      </button>

      {results.length > 0 && (
  <div className="mt-5">
    <h2 className="text-2xl mb-3">Results:</h2>
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 shadow overflow-hidden border-b border-gray-200 sm:rounded-lg">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Support</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Total Rules</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Frequent Itemsets</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rules</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {results.map((result, index) => (
            <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{result.support}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{result.confidence}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{result.total_rules}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <ul className="list-disc list-inside">
                  {result.frequent_itemsets.slice(0, 5).map((itemset, itemsetIndex) => (
                    <li key={itemsetIndex} className="text-gray-500">{itemset}</li>
                  ))}
                </ul>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <ul className="list-disc list-inside">
                  {Object.entries(result.rules).map(([key, value], ruleIndex) => (
                    <li key={ruleIndex} className="text-gray-500">
                      <strong>{key}:</strong> {value}
                    </li>
                  ))}
                </ul>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
)}

      

      {error && (
        <div className="mt-5">
          <h2 className="text-2xl mb-2">Error:</h2>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}

export default Task1;
