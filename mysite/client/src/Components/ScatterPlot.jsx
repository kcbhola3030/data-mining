import React from 'react';
import Plot from 'react-plotly.js';

const ScatterPlot = ({data}) => {
  
  const trace = {
    x: Array.from({ length: data?.length }, (_, i) => i + 1),
      y: data,
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
