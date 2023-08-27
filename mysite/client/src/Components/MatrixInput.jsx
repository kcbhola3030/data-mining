import React, { useState } from "react";

const MatrixInput = () => {
  const [matrix, setMatrix] = useState([
    ["", "", ""],
    ["", "", ""],
    ["", "", ""], // Additional row for column sums
  ]);
  const [expectedMatrix, setExpectedMatrix] = useState([]);
  const [showExpectedTable, setShowExpectedTable] = useState(false);
  const [calculatedValues, setCalculatedValues] = useState([]);
  const [calculatedSum, setCalculatedSum] = useState(0);

  const headers = ["Column 1", "Column 2", "Column 3"];
  const totalLabel = "Total";

  const handleInputChange = (rowIndex, colIndex, value) => {
    const updatedMatrix = [...matrix];
    updatedMatrix[rowIndex][colIndex] = value;
    setMatrix(updatedMatrix);
  };

  // Calculate the sums of rows, columns, and total
  const calculateSums = () => {
    const updatedMatrix = [...matrix];
    const numRows = updatedMatrix.length - 1; // Exclude the last row (total row)
    const numCols = updatedMatrix[0].length - 1; // Exclude the last column (total column)

    for (let row = 0; row < numRows; row++) {
      let rowSum = 0;
      for (let col = 0; col < numCols; col++) {
        rowSum += parseFloat(updatedMatrix[row][col] || 0);
      }
      updatedMatrix[row][numCols] = rowSum.toFixed(2); // Update the row sum
    }

    for (let col = 0; col < numCols; col++) {
      let colSum = 0;
      for (let row = 0; row < numRows; row++) {
        colSum += parseFloat(updatedMatrix[row][col] || 0);
      }
      updatedMatrix[numRows][col] = colSum.toFixed(2); // Update the column sum
    }

    // Calculate the total
    let total = 0;
    for (let row = 0; row < numRows; row++) {
      total += parseFloat(updatedMatrix[row][numCols] || 0);
    }
    updatedMatrix[numRows][numCols] = total.toFixed(2); // Update the total

    setMatrix(updatedMatrix);
  };

  const calculateExpectedValues = () => {
    const numRows = matrix.length - 1;
    const numCols = matrix[0].length - 1;

    const expectedValues = [];
    for (let row = 0; row < numRows; row++) {
      const expectedRow = [];
      for (let col = 0; col < numCols; col++) {
        const expectedValue = (
          (parseFloat(matrix[row][numCols]) *
            parseFloat(matrix[numRows][col])) /
          parseFloat(matrix[numRows][numCols])
        ).toFixed(2);
        expectedRow.push(expectedValue);
      }
      expectedValues.push(expectedRow);
    }

    const rowSums = expectedValues.map((row) =>
      row.reduce((sum, value) => sum + parseFloat(value), 0)
    );
    const colSums = expectedValues[0].map((_, colIndex) =>
      expectedValues.reduce((sum, row) => sum + parseFloat(row[colIndex]), 0)
    );
    const totalSum = rowSums.reduce((sum, value) => sum + parseFloat(value), 0);

    // Add the row and column sums to the expected values matrix
    const withSums = expectedValues.map((row, rowIndex) =>
      row.concat(rowSums[rowIndex])
    );
    withSums.push([...colSums, totalSum]);

    setExpectedMatrix(withSums); // Update the expected matrix
    setShowExpectedTable(true);
  };

  const calculateCalculatedValues = () => {
    const numRows = matrix.length - 1;
    const numCols = matrix[0].length - 1;

    const calculatedValues = [];
    let sum = 0;

    for (let row = 0; row < numRows; row++) {
      const calculatedRow = [];
      for (let col = 0; col < numCols; col++) {
        const userValue = parseFloat(matrix[row][col]);
        const expectedValue = parseFloat(expectedMatrix[row][col]);
        const calculatedValue =
          (userValue - expectedValue) ** 2 / expectedValue;
        calculatedRow.push(calculatedValue.toFixed(3));
        sum += calculatedValue;
      }
      calculatedValues.push(calculatedRow);
    }

    setCalculatedValues(calculatedValues);
    setCalculatedSum(sum.toFixed(3));
  };
  return (
    <div className="p-4">
      <table className="table-auto border border-collapse">
        <thead>
          <tr>
            <th></th>
            {headers.map((header, colIndex) => (
              <th key={colIndex} className="border p-2 text-center">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th className="border p-2 text-center">
                {rowIndex < matrix.length - 1
                  ? `Row ${rowIndex + 1}`
                  : totalLabel}
              </th>
              {row.map((cell, colIndex) => (
                <td key={colIndex} className="border p-2">
                  {rowIndex < matrix.length - 1 && colIndex < row.length - 1 ? (
                    <input
                      type="text"
                      value={cell}
                      onChange={(e) =>
                        handleInputChange(rowIndex, colIndex, e.target.value)
                      }
                      onBlur={calculateSums} // Calculate sums when leaving an input field
                      className="w-full bg-gray-100 p-1 border rounded"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      {cell}
                    </div>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <button
        onClick={calculateExpectedValues}
        className="mt-4 bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
      >
        Calculate Expected Table
      </button>

      {/* {showExpectedTable && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-2">Expected Values</h2>
          <table className="table-auto border border-collapse">
            <thead>
              <tr>
                <th></th>
                {headers.map((header, colIndex) => (
                  <th key={colIndex} className="border p-2 text-center">
                    {header}
                  </th>
                ))}
                <th className="border p-2 text-center">Row Sum</th>
              </tr>
            </thead>
            <tbody>
              {expectedMatrix.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  <th className="border p-2 text-center">
                    Row {rowIndex + 1}
                  </th>
                  {row.map((value, colIndex) => (
                    <td key={colIndex} className="border p-2">
                      <div className="w-full h-full flex items-center justify-center">
                        {value}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
              <tr>
                <th className="border p-2 text-center">{totalLabel}</th>
                {expectedMatrix[0].map((_, colIndex) => (
                  <td key={colIndex} className="border p-2">
                    <div className="w-full h-full flex items-center justify-center">
                      {expectedMatrix[expectedMatrix.length - 1][colIndex]}
                    </div>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      )} */}

      {showExpectedTable && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-2">Expected Values</h2>
          <table className="table-auto border border-collapse">
            <thead>
              <tr>
                <th></th>
                {headers.map((header, colIndex) => (
                  <th key={colIndex} className="border p-2 text-center">
                    {header}
                  </th>
                ))}
                <th className="border p-2 text-center">Row Sum</th>
              </tr>
            </thead>
            <tbody>
              {expectedMatrix.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  <th className="border p-2 text-center">Row {rowIndex + 1}</th>
                  {row.map((value, colIndex) => (
                    <td key={colIndex} className="border p-2">
                      <div className="w-full h-full flex items-center justify-center">
                        {value}
                      </div>
                    </td>
                  ))}
                  <td className="border p-2 text-center">
                    {expectedMatrix[rowIndex][expectedMatrix[0].length - 1]}
                  </td>
                </tr>
              ))}
              <tr>
                <th className="border p-2 text-center">{totalLabel}</th>
                {expectedMatrix[0].map((_, colIndex) => (
                  <td key={colIndex} className="border p-2">
                    <div className="w-full h-full flex items-center justify-center">
                      {expectedMatrix[expectedMatrix.length - 1][colIndex]}
                    </div>
                  </td>
                ))}
                <td className="border p-2 text-center">
                  {
                    expectedMatrix[expectedMatrix.length - 1][
                      expectedMatrix[0].length - 1
                    ]
                  }
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <button
        onClick={calculateCalculatedValues}
        className="mt-4 bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
      >
        Calculate Chi Square
      </button>

      {calculatedValues.length > 0 && (
        <div className="mt-4">
          <h2 className="text-xl font-semibold mb-2">Calculated Values</h2>
          {calculatedValues.map((row, rowIndex) => (
            <div key={rowIndex} className="mb-2">
              Row {rowIndex + 1} Calculated Values: {row.join(", ")}
            </div>
          ))}
          <p className="mt-2 font-semibold">
            Chi Square Value: {calculatedSum}
          </p>
        </div>
      )}
    </div>
  );
};

export default MatrixInput;
