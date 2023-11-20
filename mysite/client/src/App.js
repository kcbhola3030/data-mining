import React from 'react';
import HomePage from './Pages/HomePage';
import {  Routes, Route } from "react-router-dom";
import { Dashboard } from './Pages/Dashboard';
import { Navbar } from './Components/Navbar';
import { Chi } from './Pages/Chi';
import Pearson from './Components/Pearson';
// import Normalization from './Pages/Normalization';
import CSVUploadAndDisplay from './Components/CSVUploadAndDisplay';
import Classifier from './Pages/Classifier';
import OtherClassifier from './Pages/OtherClassifier';
import DendrogramDisplay from './Components/DendrogramDisplay';
import Kmeans from './Pages/Kmeans';
import PageRank from './Components/PageRank';
import HitScores from './Components/HitScores';
import Crawl from './Components/Crawl';
import Vkmeans from './Components/Vkmeans';
import Kmedoids from './Components/Kmedoids';
import BIRCH from './Components/BIRCH';
import DBSCAN from './Components/DBSCAN';
import ClusterValidation from './Components/ClusterValidation';
import ClusteringForm from './Components/ClusteringForm';
import Clusters from './Components/Clusters';
import Assign8 from './Pages/Assign8';
import Task1 from './Components/Task1';


function App() {
  return (
<>
      <Navbar />
      <div className="App pt-16"> {/* Add padding top to push content below navbar */}
        <Routes>
          <Route exact path="/" element={<HomePage />} />
          <Route exact path="/dashboard" element={<Dashboard />} />
          <Route exact path="/chi" element={<Chi />} />
          <Route exact path="/pearson" element={<Pearson />} />
          <Route exact path="/normalization" element={<CSVUploadAndDisplay/>} />
          <Route exact path="/classifier" element={<Classifier/>} />
          <Route exact path="/otherClassifier" element={<OtherClassifier/>} />
          <Route exact path="/dendrogram" element={<DendrogramDisplay/>} />
          <Route exact path="/kmeans" element={<Kmeans/>} />
          <Route exact path="/pagerank" element={<PageRank/>} />
          <Route exact path="/hitscore" element={<HitScores/>} />
          <Route exact path="/crawl" element={<Crawl/>} />
          <Route exact path="/clusters" element={<Clusters/>} />
          <Route exact path="/clusters/vkmeans" element={<Vkmeans/>} />
          <Route exact path="/clusters/kmedoids" element={<Kmedoids/>} />
          <Route exact path="/clusters/birch" element={<BIRCH/>} />
          <Route exact path="/clusters/dbscan" element={<DBSCAN/>} />
          <Route exact path="/clusters/clustervalidation" element={<ClusteringForm/>} />     
          <Route exact path="/assign7" element={<Task1/>} />     
          <Route exact path="/assign8" element={<Assign8/>} />     
        </Routes>
      </div>
    </>
  );
}

export default App;
