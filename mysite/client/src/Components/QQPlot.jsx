import React from 'react';
import Plot from 'react-plotly.js';

const QQPlot = ({data}) => {
  // const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // Sample data

  const trace = {
    y:data,
    type: 'scatter',
    mode: 'markers',
    name: 'Q-Q Plot',
  };

  return (
    <div>
      <h2>Q-Q Plot</h2>
      <Plot data={[trace]} />
    </div>
  );
};

export default QQPlot;
