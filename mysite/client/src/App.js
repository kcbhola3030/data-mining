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
          
        </Routes>
      </div>
    </>
  );
}

export default App;
