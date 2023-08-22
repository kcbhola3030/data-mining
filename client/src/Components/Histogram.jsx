import React from 'react';
import Plot from 'react-plotly.js';

const Histogram = ({data}) => {
//   const data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6]; // Sample data

  const trace = {
    x: data,
    type: 'histogram',
  };

  return (
    <div>
      <h2>Histogram</h2>
      <Plot data={[trace]} />
    </div>
  );
};

export default Histogram;
