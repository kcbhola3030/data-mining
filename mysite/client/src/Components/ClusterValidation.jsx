import React, { useState } from 'react';
import axios from 'axios';

const ClusterValidation = () => {
  const [dataset, setDataset] = useState('');
  const [algorithm, setAlgorithm] = useState('');
  const [kValue, setKValue] = useState('');
  const [result, setResult] = useState('');
  const [resultsTable, setResultsTable] = useState([]);

  const handleRunAlgorithm = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/clustering/', {
        dataset,
        algorithm,
        k_value: kValue,
      });

      if (algorithm === 'agnes') {
        // Handle dendrogram image response
        setResult(
          <img
            src={`data:image/png;base64, ${response.data.image}`}
            alt="Dendrogram"
          />
        );
      } else {
        // Handle other algorithm results
        const resultText = `ARI Score: ${response.data['ARI Score']}, Silhouette Score: ${response.data['Silhouette Score']}`;
        setResult(resultText);

        // Update results table
        updateResultsTable(algorithm, response.data['ARI Score'], response.data['Silhouette Score']);
      }
    } catch (error) {
      console.error('Error running clustering algorithm:', error);
    }
  };

  const updateResultsTable = (algorithm, ariScore, silhouetteScore) => {
    // Send a POST request to update the backend results_df
    axios.post('http://127.0.0.1:8000/update_results/', {
      algorithm,
      ari_score: ariScore,
      silhouette_score: silhouetteScore,
    })
    .then(() => {
      console.log('Results table updated successfully');
      // Fetch updated results table after the update
      fetchResultsTable();
    })
    .catch((error) => console.error('Error updating results table:', error));
  };

  const fetchResultsTable = () => {
    // Fetch the updated results table
    axios.get('http://127.0.0.1:8000/get_results_table/')
      .then((response) => setResultsTable(response.data))
      .catch((error) => console.error('Error fetching results table:', error));
  };

  return (
    <div>
      {/* ... (existing JSX code) */}

      <div>
        {result && (
          <div>
            <h3>Algorithm Result:</h3>
            {result}
          </div>
        )}
      </div>

      {/* Display results table */}
      <div>
        <h2>Results Table:</h2>
        <table>
          <thead>
            <tr>
              <th>Algorithm</th>
              <th>ARI Score</th>
              <th>Silhouette Score</th>
            </tr>
          </thead>
          <tbody>
            {/* Map through the rows of resultsTable and display each row */}
            {resultsTable.map((row, index) => (
              <tr key={index}>
                <td>{row.Algorithm}</td>
                <td>{row['ARI Score']}</td>
                <td>{row['Silhouette Score']}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ClusterValidation;
