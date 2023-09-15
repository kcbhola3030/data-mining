import React, { useState } from 'react';
import axios from 'axios';

function OtherClassifier() {
  const [selectedClassifier, setSelectedClassifier] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [results, setResults] = useState(null);

  const handleClassifierChange = (event) => {
    setSelectedClassifier(event.target.value);
  };

  const handleDatasetChange = (event) => {
    setSelectedDataset(event.target.value);
  };

//   const handleSubmit = async () => {
//     try {
//         // Make an API request to your backend with selectedClassifier and selectedDataset
//         const response = await axios.post('/api/your-endpoint', {
//           classifier: selectedClassifier,
//           dataset: selectedDataset,
//         });
  
//         // Once you have the response, you can set the results in state using setResults
//         setResults(response.data);
//       } catch (error) {
//         // Handle any errors here, e.g., display an error message to the user
//         console.error('API request failed:', error);
//       }
//     const exampleResponse = {
//       confusionMatrix: [
//         [50, 5],
//         [10, 85],
//       ],
//       accuracy: 0.9,
//       misclassificationRate: 0.1,
//       sensitivity: 0.89,
//       specificity: 0.91,
//       precision: 0.91,
//       recall: 0.89,
//     };

//     setResults(exampleResponse);
//   };

// const handleSubmit = async () => {
//     if (!selectedClassifier || !selectedDataset) {
//       return; // Prevent making the request if classifier or dataset is not selected
//     }
  
//     try {
//       const response = await fetch(`http://127.0.0.1:8000/${selectedClassifier.toLowerCase()}/`, {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ dataset: selectedDataset }),
//       });
//       console.log(typeof(selectedDataset))
//       if (!response.ok) {
//         throw new Error('Network response was not ok');
//       }
  
//       const responseData = await response.json();
//       console.log(responseData,response)
//       setResults(responseData);
//     } catch (error) {
//       console.error('Error:', error);
//       // Handle error state or display an error message to the user
//     }
//   };

const handleSubmit = async () => {
    if (!selectedClassifier || !selectedDataset) {
      return; // Prevent making the request if classifier or dataset is not selected
    }
  
    try {
      const response = await axios.post(`http://127.0.0.1:8000/${selectedClassifier.toLowerCase()}/`, {
        dataset: selectedDataset,
      });
  
      if (response.status !== 200) {
        throw new Error('Network response was not ok');
      }
  
      const responseData = response.data;
      setResults(responseData);
    } catch (error) {
      console.error('Error:', error);
      // Handle error state or display an error message to the user
    }
  };
  
  

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-semibold mb-4">Classifier Dashboard</h1>
      <div className="mb-4">
        <label htmlFor="classifier" className="block font-semibold">
          Select Classifier:
        </label>
        <select
          id="classifier"
          className="w-full border p-2 rounded"
          onChange={handleClassifierChange}
          value={selectedClassifier}
        >
          <option value="">Select Classifier</option>
          <option value="regression">Regression Classifier</option>
          <option value="naive_bayesian">Naïve Bayesian Classifier</option>
          <option value="knn">k-NN Classifier</option>
          <option value="ann">Three-layer ANN Classifier</option>
        </select>
      </div>
      <div className="mb-4">
        <label htmlFor="dataset" className="block font-semibold">
          Select Dataset:
        </label>
        <select
          id="dataset"
          className="w-full border p-2 rounded"
          onChange={handleDatasetChange}
          value={selectedDataset}
        >
          <option value="">Select Dataset</option>
          <option value="IRIS">IRIS Dataset</option>
          <option value="BreastCancer">Breast Cancer Dataset</option>
        </select>
      </div>
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        onClick={handleSubmit}
        disabled={!selectedClassifier || !selectedDataset}
      >
        Submit
      </button>
      {results && (
        <div className="mt-4">
          <h2 className="text-lg font-semibold mb-2">Classification Results:</h2>
          <pre className="bg-gray-100 p-4 rounded">{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default OtherClassifier;
