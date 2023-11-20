import React, { useState, useEffect } from "react";
import axios from "axios";
import ClusteringResult from "./ClusteringResult";
// import Ass6Nav from "./Ass6Nav";

function ClusteringForm() {
  const [dataset, setDataset] = useState("iris");
  const [algorithm, setAlgorithm] = useState("agnes");
  const [kValue, setKValue] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [ariScore, setAriScore] = useState(null);
  const [silScore, setSilScore] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();

    axios
      .post("http://127.0.0.1:8000/clustering/", {
        dataset,
        algorithm,
        k_value: kValue,
      })
      .then((response) => {
        if (algorithm === "agnes" || algorithm === "diana") {
          setImageUrl(response.data.image);
        } else {
          setAriScore(response.data["ARI Score"]);
          setSilScore(response.data["Silhouette Score"]);
        }
      })
      .catch((error) => {
        console.error("Error fetching clustering results: ", error);
      });
  };

  return (
    <div className="container mx-auto p-4">
        {/* <Ass6Nav /> */}
      <form onSubmit={handleSubmit}>
        {/* ... (other form inputs) */}
        <label className="block mb-2">
          Enter Value of K:
          <br />
          <input
            type="number"
            value={kValue}
            onChange={(e) => setKValue(e.target.value)}
            className="w-full border rounded py-2 px-3"
          />
        </label>
        <br />

        <label className="block mb-2">
          Select Dataset:
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full border rounded py-2 px-3"
          >
            <option value="iris">IRIS</option>
            <option value="breast_cancer">Breast Cancer</option>
          </select>
        </label>
        

        <label className="block mb-2">
          Select Algorithm:
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="w-full border rounded py-2 px-3"
          >
            <option value="agnes">AGNES</option>
            <option value="diana">DIANA</option>
            <option value="kmeans">k-Means</option>
            <option value="kmedoids">k-Medoids</option>
            <option value="birch">BIRCH</option>
            <option value="dbscan">DBSCAN</option>
          </select>
        </label>

        <button
          type="submit"
          className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
        >
          Submit
        </button>
      </form>
      {/* C:\Users\Lenovo\Desktop\Vishal\DM\kvishal\data-mining\mysite\clustering_result.png */}
      {algorithm === "agnes" || algorithm === "diana" ? (
        <ClusteringResult algorithm={algorithm} imageUrl={imageUrl} />
      ) : (
        <ClusteringResult
          algorithm={algorithm}
          ariScore={ariScore}
          silScore={silScore}
        />
      )}
    </div>
  );
}

export default ClusteringForm;
