import React, { useEffect, useState } from "react";
import axios from "axios";
import dendrogram from '../../src/dendrogram.png'

function Kmeans() {
  const [imageUrl, setImageUrl] = useState('');
  const [dataset, setDataset] = useState('');
  const [kval, setKval] = useState('');

  const handleDatasetChange = (e) => {
    setDataset(e.target.value);
  };

  const handleMethodChange = (e) => {
    setKval(e.target.value);
    console.log(kval)
  };

  const handleSubmit = (e) => {
    e.preventDefault();


    axios.post("http://127.0.0.1:8000/kmeans/", { 
  dataset:dataset,
  k:kval
 })
  .then((response) => {
      
console.log(kval,dataset)
    setImageUrl(response.data)
    })
  .catch((error) => {
    console.error("Error fetching dendrogram image: ", error);
  });
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-semibold mb-4">Kmeans </h1>
      <label className="block mb-2">
        Enter Value of K:
        <br />
        <input type="number"  className="w-full border rounded py-2 px-3"/>
      </label>
      <br />

      <label className="block mb-2">
        Select Method:
        <select
          value={dataset}
          onChange={handleDatasetChange}
          className="w-full border rounded py-2 px-3"
        >
           <option value="IRIS">IRIS</option>
            <option value="BreastCancer">Breast Cancer</option>
          
        </select>
      </label>
      
      <button
        onClick={handleSubmit}
        className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
      >
        Submit
      </button>
    
      {imageUrl && (
        <div>
          <h2>Dendrogram:</h2>
          <img src={dendrogram} alt="Decision Tree" />
        </div>
      )}
    </div>
  );
}

export default Kmeans;
