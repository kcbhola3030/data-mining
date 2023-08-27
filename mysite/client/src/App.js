import React from 'react';
import HomePage from './Pages/HomePage';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Dashboard } from './Pages/Dashboard';
import { Navbar } from './Components/Navbar';
import { Chi } from './Pages/Chi';
function App() {
  return (
<>
{/* <Navbar/> */}
<BrowserRouter>
      <Routes>
      {/* <Route path="/" element={<Navbar />}/> */}
      <Route exact path="/" element={<HomePage />}/>
      <Route exact path="/dashboard" element={<Dashboard />}/>
      <Route exact path="/chi" element={<Chi />}/>
       
      </Routes>
    </BrowserRouter>
    <div className="App">
    </div>
</>
  );
}

export default App;
