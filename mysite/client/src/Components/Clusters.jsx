import React from 'react'
// import Ass6Nav from './Ass6Nav'
import { Link } from 'react-router-dom'


function Clusters() {
  return (
    <>
 <div>
      <div style={{ display: 'flex', justifyContent: 'center', margin: '20px' }}>
        <strong>Clusters</strong>
      </div>
      <div className="flex justify-center h-200">
        <div className="bg-gray-200 p-4 rounded shadow-md">
          <ul className="flex flex-col space-y-2">
            <li>
              <Link to='/clusters/vkmeans' className="block py-2 px-4 text-gray-900 rounded hover:bg-gray-300">KMeans</Link>
            </li>
            <li>
              <Link to='/clusters/kmedoids' className="block py-2 px-4 text-gray-900 rounded hover:bg-gray-300">KMedoids</Link>
            </li>
            <li>
              <Link to='/clusters/birch' className="block py-2 px-4 text-gray-900 rounded hover:bg-gray-300">BIRCH</Link>
            </li>
            <li>
              <Link to='/clusters/dbscan' className="block py-2 px-4 text-gray-900 rounded hover:bg-gray-300">DBSCAN</Link>
            </li>
            <li>
              <Link to='/clusters/clustervalidation' className="block py-2 px-4 text-gray-900 rounded hover:bg-gray-300">Cluster Validation</Link>
            </li>
            {/* Add other cluster types here with appropriate Link components */}
          </ul>
        </div>
      </div>
    </div>
    </>

  )
}

export default Clusters