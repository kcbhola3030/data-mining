import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import "./HomePage.css";
import home from "./home.jpg";
import { Dashboard } from "./Dashboard";
import axios from 'axios'

const HomePage = () => {
  const [showDashboard,setShowDashboard]=useState(false);
  const [fileSelected, setFileSelected] = useState(false);
  const [data,setData] = useState({})
  const [csvData,setCsvData] = useState({})
  const file=[];




  const handleFileChange = (event) => {
    if (event.target.files.length > 0) {
      setFileSelected(true);
    } else {
      setFileSelected(false);
    }
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    if (file) {
      const parsedData = await parseCSV(file);
      setCsvData(parsedData.data);
    }

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/upload-csv/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Content-Disposition': `attachment; filename="${file.name}"`,
        },
      });
      alert(response.data.statistics)
      console.log('File uploaded:', response.data);
      setData(response.data.statistics)
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const parseCSV = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target.result;
        const parsedData = result.split('\n').map((row) => row.split(','));
        resolve({ data: parsedData });
      };
      reader.onerror = (error) => {
        reject(error);
      };
      reader.readAsText(file);
    });
  };
  return (
    <>
      <div className="flex flex-col h-screen">
        <div className="flex justify-around items-center">
          <div className="bg-white p-4 w-1/2">
            <h1 className="text-5xl font-bold leading-12">
              Exploring Statistics <br></br>
              Metrics
            </h1>
            <p className="mt-4 text-gray-500 leading-7">
              Welcome to our comprehensive guide on key statistics metrics! In
              this exploration, we delve into fundamental concepts such as mean,
              mode, median, quartile, interquartile range, midrange, variance,
              and standard deviation. Understanding these metrics is essential
              for gaining insights from data, making informed decisions, and
              uncovering patterns within datasets. Whether you're new to
              statistics or looking to deepen your knowledge, our guide will
              help you navigate these essential measures with clarity and
              confidence.
            </p>

            <label
              class="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              for="file_input"
            >
              Upload file
            </label>
            <input
              onChange={handleFileChange}
              class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
              aria-describedby="file_input_help"
              id="file_input"
              type="file"
            />
            <p
              class="mt-1 text-sm text-gray-500 dark:text-gray-300"
              id="file_input_help"
            >
              Upload your CSV here
            </p>
            <p>
          {fileSelected ? (
            <>
              <button
              onClick={(e)=>{


                setShowDashboard(true)}
              }
              
                class="text-white bg-gray-800 hover:bg-gray-900 focus:outline-none focus:ring-4 focus:ring-gray-300 font-medium rounded-lg text-sm px-5 py-2.5 mr-2 mb-2 dark:bg-gray-800 dark:hover:bg-gray-700 dark:focus:ring-gray-700 dark:border-gray-700"
              >
                Get Started
              </button>
            </>
          ) : (
            <></>
          )}
        </p>
          </div>
          <img src={home} className="w-2/5" alt="home" />
        </div>
        {showDashboard?<Dashboard data={data}/>:<></>}
      </div>
    </>
  );
};

export default HomePage;
