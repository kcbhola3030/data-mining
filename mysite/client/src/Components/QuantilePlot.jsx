import React from "react";
import Plot from "react-plotly.js";

const QuantilePlot = ({ data }) => {
  // const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // Sample data

  const trace = {
    x:data,
    type: "scatter",
    mode: "markers",
    name: "Quantile Plot",
  };

  return (
    <div>
      <h2>Quantile Plot</h2>
      <Plot data={[trace]} />
    </div>
  );
};

export default QuantilePlot;
