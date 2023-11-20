import React from 'react'
// import Ass6Nav from './Ass6Nav'
import { Link } from 'react-router-dom'


function Assign8() {
  return (
    <>
 <div>
      <div style={{ display: 'flex', justifyContent: 'center', margin: '20px' }}>
        <strong>Crawlers</strong>
      </div>
      <div className="flex justify-center h-200">
        <div className="bg-gray-200 p-4 rounded shadow-md">
          <ul className="flex flex-col space-y-2">
            <li>
              <Link to='/pagerank' className="block py-2 px-4 text-gray-900 rounded hover:bg-gray-300">Page Rank</Link>
            </li>
            <li>
              <Link to='/crawl' className="block py-2 px-4 text-gray-900 rounded hover:bg-gray-300">Crawler</Link>
            </li>
            <li>
              <Link to='/hitscore' className="block py-2 px-4 text-gray-900 rounded hover:bg-gray-300">Hit Score</Link>
            </li>
           
          </ul>
        </div>
      </div>
    </div>
    </>

  )
}

export default Assign8