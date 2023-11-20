// BIRCH.js

import React, { useState } from "react";
import axios from "axios";
import birchImage1 from '../birchOf_IRIS.png';  // Update with the correct image path
import birchImage2 from '../birchOf_BreastCancer.png';
// import Ass6Nav from "./Ass6Nav";

function BIRCH() {
  const [imageUrl, setImageUrl] = useState('');
  const [dataset, setDataset] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();

    if (dataset) {
      axios.post("http://127.0.0.1:8000/birch/", {
        dataset: dataset,
      })
      .then((response) => {
        console.log("BIRCH frontend");
        setImageUrl(response.data);
      })
      .catch((error) => {
        console.error("Error fetching BIRCH image: ", error);
      });
    } else {
      console.error("Dataset is empty. Cannot submit the form.");
    }
  };

  return (
    <div className="container mx-auto p-4">
        {/* <Ass6Nav/> */}
      <form onSubmit={handleSubmit}>
        <label className="block mb-2">
          Select Dataset:
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full border rounded py-2 px-3"
          >
            <option value="IRIS">IRIS</option>
            <option value="BreastCancer">Breast Cancer</option>
          </select>
        </label>

        <button
          type="submit"
          className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
        >
          Submit
        </button>
      </form>
      {imageUrl && dataset === 'IRIS' && (
  <div>
    <h2>BIRCH:</h2>
    <img src={birchImage1} alt="birch-iris" />
  </div>
)}

{imageUrl && dataset === 'BreastCancer' && (
  <div>
    <h2>BIRCH:</h2>
    <img src={birchImage2} alt="birch-BreastCancer" />
  </div>
)}
      
    </div>
  );
}

export default BIRCH;


