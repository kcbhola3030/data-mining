import React, { useEffect, useState } from "react";
import QuantilePlot from "../Components/QuantilePlot";
import QQPlot from "../Components/QQPlot";
import ScatterPlot from "../Components/ScatterPlot";
import BoxPlot from "../Components/BoxPlot";
import Histogram from "../Components/Histogram";

export const Dashboard = ({ data }) => {
  // const plotLayout = {
  //   margin: { t: 20, r: 20, b: 50, l: 50 },
  //   autosize: true,
  //   xaxis: {
  //     title: 'X-axis Label',
  //   },
  //   yaxis: {
  //     title: 'Y-axis Label',
  //   },
  // };
  const plotLayout = {};

  const [mean, setMean] = useState(0);
  const [median, setMedian] = useState(0);
  const [mode, setMode] = useState(0);
  const [variance, setVariance] = useState(0);
  const [std, setStd] = useState(0);
  const [mid, setMid] = useState(0);
  const [mean_cal, setMeanCal] = useState(0);
  const [median_cal, setMedianCal] = useState(0);
  const [mode_cal, setModeCal] = useState(0);
  const [variance_cal, setVarianceCal] = useState(0);
  const [std_cal, setStdCal] = useState(0);
  const [mid_cal, setMidCal] = useState(0);

  const [col, setCol] = useState([]);

  const [range, setRange] = useState(0);
  const [irange, setIRange] = useState(0);
  const [quartile, setQuartile] = useState([]);
  const [five, setFive] = useState([]);


  const [selectedKey, setSelectedKey] = useState(); // Initialize with the first key
  const handleKeyChange = (event) => {
    setSelectedKey(event.target.value);

    const finalData = data.statistics[selectedKey];
    setMean(finalData?.mean.toFixed(2));
    setMedian(finalData?.median.toFixed(2));
    setMode(finalData?.mode.toFixed(2));
    setVariance(finalData?.variance.toFixed(2));
    setStd(finalData?.std_deviation.toFixed(2));
    setMid(finalData?.mid_range.toFixed(2));

    setMeanCal(finalData?.mean_cal.toFixed(2));
    setMedianCal(finalData?.median_cal.toFixed(2));
    setModeCal(finalData?.mode_cal.toFixed(2));
    setVarianceCal(finalData?.variance_cal.toFixed(2));
    setStdCal(finalData?.std_deviation_cal.toFixed(2));
    setMidCal(finalData?.midrange_cal.toFixed(2));
    setCol(finalData?.converted_values);

    setRange(finalData?.range.toFixed(2));
    setIRange(finalData?.interquartile_range.toFixed(2));
    setQuartile(finalData?.quartiles);
    setFive(finalData?.five_number_summary);
  };

  
  return (
    <>
      {data ? <>
      
      <div class="m-0 font-sans text-base antialiased font-normal dark:bg-slate-900 leading-default bg-gray-50 text-slate-500">
        <div class="absolute w-full bg-blue-500 dark:hidden min-h-75"></div>

        <main class="relative h-full max-h-screen transition-all duration-200 ease-in-out xl:ml-68 rounded-xl">
          <div class="w-full px-6 py-6 mx-auto">
            <div class="flex flex-wrap -mx-3">
              <div class="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
                <div class="relative flex flex-col min-w-0 break-words bg-white shadow-xl dark:bg-slate-850 dark:shadow-dark-xl rounded-2xl bg-clip-border">
                  <div class="flex-auto p-4">
                    <div class="flex flex-row -mx-3">
                      <div class="flex-none w-2/3 max-w-full px-3">
                        <div>
                          <p class="mb-0 font-sans text-sm font-semibold leading-normal uppercase dark:text-dark dark:opacity-60">
                            Range
                          </p>
                          <h5 class="mb-2 font-bold dark:text-dark">{range}</h5>
                        </div>
                      </div>
                      <div class="px-3 text-right basis-1/3">
                        <div class="inline-block w-12 h-12 text-center rounded-circle bg-gradient-to-tl from-blue-500 to-violet-500">
                          <i class="ni leading-none ni-money-coins text-lg relative top-3.5 text-white"></i>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
                <div class="relative flex flex-col min-w-0 break-words bg-white shadow-xl dark:bg-slate-850 dark:shadow-dark-xl rounded-2xl bg-clip-border">
                  <div class="flex-auto p-4">
                    <div class="flex flex-row -mx-3">
                      <div class="flex-none w-2/3 max-w-full px-3">
                        <div>
                          <p class="mb-0 font-sans text-sm font-semibold leading-normal uppercase dark:text-dark dark:opacity-60">
                            InterQuartile Range
                          </p>
                          <h5 class="mb-2 font-bold dark:text-dark">
                            {irange}
                          </h5>
                        </div>
                      </div>
                      <div class="px-3 text-right basis-1/3">
                        <div class="inline-block w-12 h-12 text-center rounded-circle bg-gradient-to-tl from-red-600 to-orange-600">
                          <i class="ni leading-none ni-world text-lg relative top-3.5 text-white"></i>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
                <div class="relative flex flex-col min-w-0 break-words bg-white shadow-xl dark:bg-slate-850 dark:shadow-dark-xl rounded-2xl bg-clip-border">
                  <div class="flex-auto p-4">
                    <div class="flex flex-row -mx-3">
                      <div class="flex-none w-2/3 max-w-full px-3">
                        <div>
                          <p class="mb-0 font-sans text-sm font-semibold leading-normal uppercase dark:text-dark dark:opacity-60">
                            Quartiles
                          </p>
                          <h5 class="mb-2 font-bold dark:text-dark">
                            {quartile?.map((value, index) => (
                              <li key={index}>
                                Quartile {index + 1}: {value}
                              </li>
                            ))}
                          </h5>
                        </div>
                      </div>
                      <div class="px-3 text-right basis-1/3">
                        <div class="inline-block w-12 h-12 text-center rounded-circle bg-gradient-to-tl from-emerald-500 to-teal-400">
                          <i class="ni leading-none ni-paper-diploma text-lg relative top-3.5 text-white"></i>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="w-full max-w-full px-3 sm:w-1/2 sm:flex-none xl:w-1/4">
                <div class="relative flex flex-col min-w-0 break-words bg-white shadow-xl dark:bg-slate-850 dark:shadow-dark-xl rounded-2xl bg-clip-border">
                  <div class="flex-auto p-4">
                    <div class="flex flex-row -mx-3">
                      <div class="flex-none w-2/3 max-w-full px-3">
                        <div>
                          <p class="mb-0 font-sans text-sm font-semibold leading-normal uppercase dark:text-dark dark:opacity-60">
                            Five Number Summary
                          </p>
                          <h5 class="mb-2 font-bold dark:text-dark">
                            {five ? (
                              <>
                                <ul>
                                  <li>Minimum Value: {five[0]}</li>
                                  <li>First Quartile (Q1): {five[1]}</li>
                                  <li>Median: {five[2]}</li>
                                  <li>Third Quartile (Q3): {five[3]}</li>
                                  <li>Maximum Value: {five[4]}</li>
                                </ul>
                              </>
                            ) : (
                              <></>
                            )}
                          </h5>
                        </div>
                      </div>
                      <div class="px-3 text-right basis-1/3">
                        <div class="inline-block w-12 h-12 text-center rounded-circle bg-gradient-to-tl from-orange-500 to-yellow-500">
                          <i class="ni leading-none ni-cart text-lg relative top-3.5 text-white"></i>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div class="flex flex-wrap mt-6 -mx-3">
              <div class="w-1/2 max-w-1/2 px-3 mt-0 mb-6 lg:mb-0 lg:w-1/2 lg:flex-none">
                <div class="relative flex flex-col min-w-0 break-words bg-white border-0 border-solid shadow-xl dark:bg-slate-850 dark:shadow-dark-xl dark:bg-gray-950 border-black-125 rounded-2xl bg-clip-border">
                  <div class="p-4 pb-0 mb-0 rounded-t-4">
                    <div class="flex justify-between">
                      <h6 class="mb-2 dark:text-white">Stats by Column</h6>
                      <select
                        className="mb-2"
                        value={selectedKey}
                        onChange={handleKeyChange}
                      >
                        {Object.keys(data?.statistics).map((key) => (
                          <option key={key} value={key}>
                            {key}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div class="overflow-x-auto">
                    <table class="items-center w-full mb-4 align-top border-collapse border-gray-200 dark:border-white/40">
                      <tbody>
                        <tr>
                          <td class="p-2 align-middle bg-transparent border-b w-3/10 whitespace-nowrap dark:border-white/40">
                            <div class="flex items-center px-2 py-1">
                              <div></div>
                              <div class="ml-6">
                                <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                  Mean
                                </h6>
                              </div>
                            </div>
                          </td>

                          <td class="p-2 align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value:
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {mean}
                              </h6>
                            </div>
                          </td>
                          <td class="p-2 text-sm leading-normal align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="flex-1 text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value (in-built):
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {mean_cal}
                              </h6>
                            </div>
                          </td>
                        </tr>
                        <tr>
                          <td class="p-2 align-middle bg-transparent border-b w-3/10 whitespace-nowrap dark:border-white/40">
                            <div class="flex items-center px-2 py-1">
                              <div></div>
                              <div class="ml-6">
                                <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                  Median
                                </h6>
                              </div>
                            </div>
                          </td>

                          <td class="p-2 align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value:
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {median}
                              </h6>
                            </div>
                          </td>
                          <td class="p-2 text-sm leading-normal align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="flex-1 text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value (in-built):
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {median_cal}
                              </h6>
                            </div>
                          </td>
                        </tr>
                        <tr>
                          <td class="p-2 align-middle bg-transparent border-b w-3/10 whitespace-nowrap dark:border-white/40">
                            <div class="flex items-center px-2 py-1">
                              <div></div>
                              <div class="ml-6">
                                <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                  Mode
                                </h6>
                              </div>
                            </div>
                          </td>

                          <td class="p-2 align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value:
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {mode}
                              </h6>
                            </div>
                          </td>
                          <td class="p-2 text-sm leading-normal align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="flex-1 text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value (in-built):
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {mode_cal}
                              </h6>
                            </div>
                          </td>
                        </tr>
                        <tr>
                          <td class="p-2 align-middle bg-transparent border-b w-3/10 whitespace-nowrap dark:border-white/40">
                            <div class="flex items-center px-2 py-1">
                              <div></div>
                              <div class="ml-6">
                                <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                  Midrange
                                </h6>
                              </div>
                            </div>
                          </td>

                          <td class="p-2 align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value:
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {mid}
                              </h6>
                            </div>
                          </td>
                          <td class="p-2 text-sm leading-normal align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="flex-1 text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value (in-built):
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {mid_cal}
                              </h6>
                            </div>
                          </td>
                        </tr>{" "}
                        <tr>
                          <td class="p-2 align-middle bg-transparent border-b w-3/10 whitespace-nowrap dark:border-white/40">
                            <div class="flex items-center px-2 py-1">
                              <div></div>
                              <div class="ml-6">
                                <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                  Variance
                                </h6>
                              </div>
                            </div>
                          </td>

                          <td class="p-2 align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value:
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {variance}
                              </h6>
                            </div>
                          </td>
                          <td class="p-2 text-sm leading-normal align-middle bg-transparent border-b whitespace-nowrap dark:border-white/40">
                            <div class="flex-1 text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value (in-built):
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {variance_cal}
                              </h6>
                            </div>
                          </td>
                        </tr>
                        <tr>
                          <td class="p-2 align-middle bg-transparent border-0 w-3/10 whitespace-nowrap">
                            <div class="flex items-center px-2 py-1">
                              <div></div>
                              <div class="ml-6 mb-0">
                                <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                  Standard Deviation
                                </h6>
                              </div>
                            </div>
                          </td>

                          <td class="p-2 align-middle bg-transparent border-0 whitespace-nowrap">
                            <div class="text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value:
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {std}
                              </h6>
                            </div>
                          </td>
                          <td class="p-2 text-sm leading-normal align-middle bg-transparent border-0 whitespace-nowrap">
                            <div class="flex-1 text-center">
                              <p class="mb-0 text-xs font-semibold leading-tight dark:text-white dark:opacity-60">
                                Value (in-built):
                              </p>
                              <h6 class="mb-0 text-sm leading-normal dark:text-white">
                                {std_cal}
                              </h6>
                            </div>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Left table ends here */}
              </div>
              <div class="w-full max-w-full px-3 mt-0 lg:w-5/12 lg:flex-none">
                <Histogram data={col} layout={plotLayout} />
              </div>
            </div>

            <div class="flex flex-wrap mt-6 -mx-3">
              <div class="w-full max-w-full px-3 mt-0 lg:w-1/2 lg:flex-none">
                <div class="border-black/12.5 dark:bg-slate-850 dark:shadow-dark-xl shadow-xl relative z-20 flex min-w-0 flex-col break-words rounded-2xl border-0 border-solid bg-white bg-clip-border">
                  <div class="border-black/12.5 mb-0 rounded-t-2xl border-b-0 border-solid p-6 pt-4 pb-0">
                    <p class="mb-0 text-sm leading-normal dark:text-dark dark:opacity-60">
                      <i class="fa fa-arrow-up text-emerald-500"></i>
                    </p>
                    <QuantilePlot data={col} layout={plotLayout} />
                    <QQPlot data={col} layout={plotLayout} />
                  </div>
                  <div class="flex-auto p-4">
                    <div>
                      <canvas id="chart-line" height="300"></canvas>
                    </div>
                  </div>
                </div>
              </div>

              <div class="w-full max-w-full px-3 lg:w-5/12 lg:flex-none">
                <ScatterPlot data={col} layout={plotLayout} />
                <BoxPlot data={col} layout={plotLayout} />

                <div class="block text-start ml-12 left-0 bottom-0 absolute right-[15%] pt-5 pb-5 text-dark">
                  <div class="inline-block w-8 h-8 mb-4 text-center text-black bg-white bg-center rounded-lg fill-current stroke-none">
                    <i class="top-0.75 text-xxs relative text-slate-700 ni ni-trophy"></i>
                  </div>
                </div>

                <button
                  btn-next
                  class="absolute z-10 w-10 h-10 p-2 text-lg text-white border-none opacity-50 cursor-pointer hover:opacity-100 far fa-chevron-right active:scale-110 top-6 right-4"
                ></button>
                <button
                  btn-prev
                  class="absolute z-10 w-10 h-10 p-2 text-lg text-white border-none opacity-50 cursor-pointer hover:opacity-100 far fa-chevron-left active:scale-110 top-6 right-16"
                ></button>
              </div>
            </div>
          </div>
          {/* Add more elements here */}
        </main>
        <div fixed-plugin>
          <a
            fixed-plugin-button
            class="fixed px-4 py-2 text-xl bg-white shadow-lg cursor-pointer bottom-8 right-8 z-990 rounded-circle text-slate-700"
          >
            <i class="py-2 pointer-events-none fa fa-cog"> </i>
          </a>
        </div>
      </div>
       </> :<div className="flex items-center justify-center h-screen">
  <p className="text-center text-gray-600 text-xl">No data loaded<br></br>
  Go to Home </p>
</div>
}
    </>
  );
};
