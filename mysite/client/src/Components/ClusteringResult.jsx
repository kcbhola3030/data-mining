import React, { useState, useEffect } from "react";
import axios from "axios";
// import img1 from 'C://Users//Lenovo//Desktop//Vishal//DM//kvishal//data-mining//mysite//clustering_result.png'
import img2 from '../clustering_result.png'
// 
function ClusteringResult({ algorithm, imageUrl, ariScore, silScore }) {
  return (
    <div>
      <h2>{algorithm} Result:</h2>
      {imageUrl && <img src={img2} alt={`${algorithm} Result`} />}
      <p>ARI Score: {ariScore}</p>
      <p>Silhouette Score: {silScore}</p>
    </div>
  );
}

export default ClusteringResult;
