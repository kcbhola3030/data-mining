import React, { useState } from 'react';
import axios from 'axios';

const Crawl = () => {
  const [url, setUrl] = useState('');
  const [showResults, setShowResults] = useState(false);
  const [bfsResult, setBfsResult] = useState([]);
  const [dfsResult, setDfsResult] = useState([]);

  const handleCrawl = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/crawl/', { url:url });
      setBfsResult(response.data.bfs_result);
      setDfsResult(response.data.dfs_result);
      setShowResults(true);
      console.log(response.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  return (
    <div className="container mx-auto">
      <input
        type="text"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        placeholder="Enter URL"
        className="m-2 p-2 border border-gray-300"
      />
      <button onClick={handleCrawl} className="m-2 p-2 bg-blue-500 text-white rounded">
        Crawl URLs
      </button>

      {showResults && (
        <div className="mt-4">
          <h2 className="text-xl font-bold">BFS Results:</h2>
          <ul>
            {bfsResult.map((link, index) => (
              <li key={index}>{link}</li>
            ))}
          </ul>

          <h2 className="text-xl font-bold mt-4">DFS Results:</h2>
          <ul>
            {dfsResult.map((link, index) => (
              <li key={index}>{link}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Crawl;
