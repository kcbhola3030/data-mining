import React from 'react';
import HomePage from './Pages/HomePage';
import {  Routes, Route } from "react-router-dom";
import { Dashboard } from './Pages/Dashboard';
import { Navbar } from './Components/Navbar';
import { Chi } from './Pages/Chi';
import Pearson from './Components/Pearson';

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
        </Routes>
      </div>
    </>
  );
}

export default App;
