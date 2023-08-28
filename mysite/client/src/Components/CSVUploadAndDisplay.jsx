import React, { useState } from "react";
import axios from "axios";

const CSVUploadAndDisplay = () => {
  const [csvFile, setCsvFile] = useState(null);
  const [columnNames, setColumnNames] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [columnData, setColumnData] = useState({});
  const [data,setData] = useState([])

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setCsvFile(file);

    if (file) {
      const reader = new FileReader();
      reader.onload = handleFileRead;
      reader.readAsText(file);
    }
  };

  const handleFileRead = (event) => {
    const csvData = event.target.result;
    const rows = csvData.split("\n");
    const header = rows[0].split(",");

    setColumnNames(header);

    const data = {};
    header.forEach((colName) => {
      data[colName] = [];
    });

    for (let i = 1; i < rows.length; i++) {
      const values = rows[i].split(",");
      header.forEach((colName, colIndex) => {
        data[colName].push(values[colIndex]);
      });
    }

    setColumnData(data);
  };


  const handleCalculate = () => {
  
    setData(columnData[selectedColumns[0]].map(Number))

  };

  const handleColumnSelect = (event) => {
    const selectedCols = Array.from(
      event.target.selectedOptions,
      (option) => option.value
    );
    setSelectedColumns(selectedCols);
  };

  const [selectedMethod, setSelectedMethod] = useState('');
  const [normalizedValues, setNormalizedValues] = useState('');



  const handleMethodChange = (event) => {
    setSelectedMethod(event.target.value);
  };
  const handleNormalize = () => {
    const values = data.map((element) => {
        const num = Number(element);
        return isNaN(num) ? 0 : num;
      });  
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
      <input
        type="file"
        onChange={handleFileChange}
        className="mb-4 border rounded p-2"
      />
      <div className="mb-4">
        <h2 className="text-xl font-semibold">Column Names:</h2>
        <ul className="list-disc ml-6">
          {columnNames.map((colName, index) => (
            <li key={index}>{colName}</li>
          ))}
        </ul>
      </div>
  

      <div className="mb-4">
        <h2 className="text-xl font-semibold">Select Column:</h2>
        <select
          className="border rounded p-2 w-full"
          onChange={handleColumnSelect}
        >
          {columnNames.map((colName, index) => (
            <option key={index} value={colName}>
              {colName}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={handleCalculate}
        className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
      >
        Calculate
      </button>



      <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">Normalization</h2>
     
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
    </div>
  );
};

export default CSVUploadAndDisplay;
