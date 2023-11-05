import React, { useEffect, useState } from "react";
import axios from "axios";
import dendrogram from '../../src/dendrogram.png'

function DendrogramDisplay() {

  const [dataset, setDataset] = useState('');
  const [method, setMethod] = useState('');

  const handleDatasetChange = (e) => {
    setDataset(e.target.value);
  };

  const handleMethodChange = (e) => {
    setMethod(e.target.value);
    console.log(method)
  };

  const handleSubmit = (e) => {
    e.preventDefault();


    axios.post("http://127.0.0.1:8000/dendrogram/", { 
  dataset:dataset,
  clustering_method:method
 })
  .then((response) => {
      console.log(method)
    })
  .catch((error) => {
    console.error("Error fetching dendrogram image: ", error);
  });
    //onSelect(dataset, method);
  };
  // useEffect(() => {
  //   // Make a GET request to the Django endpoint that serves the dendrogram image
  //   axios.get("/path-to-django-view/iris/agnes/image/")
  //     .then((response) => {
  //       const imageBase64 = response.data;
  //       setImageSrc(`data:image/png;base64, ${imageBase64}`);
  //     })
  //     .catch((error) => {
  //       console.error("Error fetching dendrogram image: ", error);
  //     });
  // }, []);

//   useEffect(() => {
  
// }, []);
  return (
    // <div className="container mx-auto p-4">
    //    <div className="max-w-md mx-auto p-4 space-y-4">
    //   <h2 className="text-xl font-semibold">Dendrogram Options</h2>
    //   <form onSubmit={handleSubmit}>
    //     <div className="mb-4">
    //       <label htmlFor="dataset" className="block text-gray-700">Select Dataset:</label>
    //       <select
    //         id="dataset"
    //         value={dataset}
    //         onChange={handleDatasetChange}
    //         className="block w-full p-2 border border-gray-300 rounded"
    //       >
    //         <option value="IRIS">IRIS</option>
    //         <option value="BreastCancer">Breast Cancer</option>
    //       </select>
    //     </div>
    //     <div className="mb-4">
    //       <label htmlFor="method" className="block text-gray-700">Select Clustering Method:</label>
    //       <select
    //         id="method"
    //         value={method}
    //         onChange={handleMethodChange}
    //         className="block w-full p-2 border border-gray-300 rounded"
    //       >
    //         <option value="agnes">AGNES</option>
    //         <option value="diana">DIANA</option>
    //       </select>
    //     </div>
    //     <button type="submit" className="w-full bg-blue-500 text-white p-2 rounded">
    //       Generate Dendrogram
    //     </button>
    //   </form>
    // </div>
    //   {dendrogram && <img  src={dendrogram} alt="Dendrogram" />}
   
    // </div>





    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-semibold mb-4">Classifier Page</h1>
      <label className="block mb-2">
        Select Method:
        <select
          value={method}
          onChange={handleMethodChange}
          className="w-full border rounded py-2 px-3"
        >
           <option value="agnes">AGNES</option>
            <option value="diana">DIANA</option>
          
        </select>
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
    
      {dendrogram && (
        <div>
          <h2>Dendrogram:</h2>
          <img src={dendrogram} alt="Decision Tree" />
        </div>
      )}
    </div>
  );
}

export default DendrogramDisplay;
