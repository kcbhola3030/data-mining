import React, { useEffect, useState } from "react";
import axios from "axios";
import kmeans1 from '../kmeansOf_IRIS.png'
import kmeans2 from '../kmeansOf_BreastCancer.png'
// import Ass6Nav from "./Ass6Nav";

function Vkmeans() {
    const [imageUrl, setImageUrl] = useState('');
    const [dataset, setDataset] = useState('');
    const [kval, setKval] = useState('');
  
    // ... (other functions)
  
    const handleSubmit = (e) => {
      e.preventDefault();
  
      // Ensure that dataset and kval are not empty before making the POST request
      if (dataset && kval) {
        axios.post("http://127.0.0.1:8000/kmeans/", {
          dataset: dataset,
          k_value: kval
        })
        .then((response) => {
          setImageUrl(response.data);
        })
        .catch((error) => {
          console.error("Error fetching dendrogram image: ", error);
        });
      } else {
        console.error("Dataset or k_value is empty. Cannot submit the form.");
      }
    };
  
    return (
      <div className="container mx-auto p-4">
        {/* ... (other JSX) */}
        <label className="block mb-2">
        {/* <Ass6Nav/> */}
          Enter Value of K:
          <br />
          <input
            type="number"
            value={kval}
            onChange={(e) => setKval(e.target.value)}
            className="w-full border rounded py-2 px-3"
          />
        </label>
        <br />
  
        <label className="block mb-2">
          Select Method:
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
            onClick={handleSubmit}
            className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
        >
            Submit
        </button>
  
        {imageUrl && dataset==='IRIS' && (
    <div>
    <h2>kmeans:</h2>
    {/* <h3>{imageUrl}</h3> */}

    <img src={kmeans1} alt="kmeans-iris" />
    </div>
)}

{imageUrl && dataset==='BreastCancer' && (
    <div>
    <h2>kmeans:</h2>
    {/* <h3>{imageUrl}</h3> */}
    <img src={kmeans2} alt="kmeans-BreastCancer" />
    </div>
)}
 
        
      </div>
    );
  }
  
  export default Vkmeans;


  
