import React from 'react';
import Plot from 'react-plotly.js';

const ScatterPlot = () => {
  const xData = [1, 2, 3, 4, 5];
  const yData = [3, 7, 1, 9, 5]; // Sample data

  const trace = {
    x: xData,
    y: yData,
    mode: 'markers',
    type: 'scatter',
  };

  return (
    <div>
      <h2 className='text-dark'> Scatter Plot</h2>
      <Plot data={[trace]} />
    </div>
  );
};

export default ScatterPlot;
