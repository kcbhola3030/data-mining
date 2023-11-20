import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Import Axios

const PageRank = () => {
  const [topPages, setTopPages] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/pagerank'); // Use Axios for GET request
        setTopPages(response.data.top_pages);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="container mx-auto">
      <h1 className="text-2xl font-bold mb-4">Top Pages by PageRank</h1>
      <div className="grid grid-cols-2 gap-4">
        {topPages.map((page, index) => (
          <div key={index} className="bg-gray-200 p-4 rounded">
            <p>Page: {page.Page}</p>
            <p>PageRank: {page.PageRank}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PageRank;
