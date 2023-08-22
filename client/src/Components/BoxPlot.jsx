import React from 'react';
import Plot from 'react-plotly.js';

const BoxPlot = () => {
  const data = [
    [1, 2, 3, 4, 5],
    [2, 3, 5, 6, 7],
    [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6],
  ]; // Sample data

  const trace = {
    y: data,
    type: 'box',
  };

  return (
    <div>
      <h2>Box Plot</h2>
      <Plot data={[trace]} />
    </div>
  );
};

export default BoxPlot;
