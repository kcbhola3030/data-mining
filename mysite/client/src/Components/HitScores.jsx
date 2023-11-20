import React, { useState, useEffect } from 'react';
import axios from 'axios';

const HitScores = () => {
  const [topAuthorityPages, setTopAuthorityPages] = useState([]);
  const [topHubPages, setTopHubPages] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const authorityResponse = await axios.get('http://127.0.0.1:8000/hitscore');
        const hubResponse = await axios.get('http://127.0.0.1:8000/hitscore');

        setTopAuthorityPages(authorityResponse.data.top_authority_pages);
        setTopHubPages(hubResponse.data.top_hub_pages);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="container mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-4">Top Authority Pages</h1>
        <div className="grid grid-cols-2 gap-4">
          {topAuthorityPages.map((page, index) => (
            <div key={index} className="bg-gray-200 p-4 rounded">
              <p>Page: {page.Page}</p>
              <p>Authority Score: {page.AuthorityScore}</p>
            </div>
          ))}
        </div>
      </div>
      <div>
        <h1 className="text-2xl font-bold mb-4">Top Hub Pages</h1>
        <div className="grid grid-cols-2 gap-4">
          {topHubPages.map((page, index) => (
            <div key={index} className="bg-gray-200 p-4 rounded">
              <p>Page: {page.Page}</p>
              <p>Hub Score: {page.HubScore}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HitScores;
