import React, { useState } from 'react';
import axios from 'axios';

const Pearson = () => {
    const [csvFile, setCsvFile] = useState(null);
    const [columnNames, setColumnNames] = useState([]);
    const [selectedColumns, setSelectedColumns] = useState([]);
    const [columnData, setColumnData] = useState({});
    const [response, setResponse] = useState(null);
    
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
        const rows = csvData.split('\n');
        const header = rows[0].split(',');
        
        setColumnNames(header);
        
        const data = {};
        header.forEach((colName) => {
            data[colName] = [];
        });
        
        for (let i = 1; i < rows.length; i++) {
            const values = rows[i].split(',');
            header.forEach((colName, colIndex) => {
                data[colName].push(values[colIndex]);
            });
        }
        
        setColumnData(data);
    };
    
    //   const handleColumnSelect = (event) => {
        //     const selectedCols = event.target.value.split(',');
        //     setSelectedColumns(selectedCols);
        //   };
        
        const handleCalculate = () => {
            
            const requestData = {
                array1: columnData[selectedColumns[0]].map(Number),
                array2: columnData[selectedColumns[1]].map(Number),
              };
            axios
            .post('http://127.0.0.1:8000/pearson', requestData)
            .then((response) => {
                setResponse(response.data);
            })
            .catch((error) => {
                console.error('Error calculating:', error);
            });
        };
        
        const handleColumnSelect = (event) => {
            const selectedCols = Array.from(event.target.selectedOptions, (option) => option.value);
            setSelectedColumns(selectedCols);
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
      {/* <div className="mb-4">
        <h2 className="text-xl font-semibold">Select Columns:</h2>
        <input
          type="text"
          placeholder="Enter column names separated by commas"
          className="border rounded p-2 w-full"
          onChange={handleColumnSelect}
        />
      </div> */}

      <div className="mb-4">
  <h2 className="text-xl font-semibold">Select Columns:</h2>
  <select
    multiple
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
      {response && (
        <div className="mt-4">
          <h2 className="text-xl font-semibold">Pearson Coefficient:</h2>
          <pre className="bg-gray-100 p-4 rounded">{response.pearson_coefficient}</pre>
        </div>
      )}
    </div>
  );
};

export default Pearson;
