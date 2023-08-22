import React from 'react';
import HomePage from './Pages/HomePage';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Dashboard } from './Pages/Dashboard';
function App() {
  return (
<>

<BrowserRouter>
      <Routes>
      <Route exact path="/" element={<HomePage />}/>
      <Route exact path="/dashboard" element={<Dashboard />}/>
       
      </Routes>
    </BrowserRouter>
    <div className="App">
    </div>
</>
  );
}

export default App;
